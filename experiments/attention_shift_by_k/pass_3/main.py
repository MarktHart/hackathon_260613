"""pass_3 — a REAL QK attention circuit that *computes* shift-by-k.

Unlike a hand-painted score matrix, here the shift-by-k pattern EMERGES from an
honest attention computation:  scores = (X W_Q) (X W_K)^T,  attn = softmax(scores).
The only hand-set pieces are the projection matrices W_Q / W_K and the positional
embedding — exactly the "base_model.py plus a few hand-set projections" delta the
conventions ask for. Nothing writes mass onto the target key directly; the
off-diagonal band falls out of the dot products.

Architecture (small delta from base_model.py — a single attention layer, no MLP):
  * residual stream X of width d_model = 2*L, split into two blocks:
      - block A (first L dims)  : a real token embedding  (consumes the token IDs)
      - block B (last  L dims)  : one-hot positional embedding  P[i] = e_i
  * H = 5 heads, head h dedicated to offset k_h = K_SWEEP[h].
      - W_Q^h reads ONLY the positional block -> Q[i] = e_i
      - W_K^h reads ONLY the positional block, shifted by k -> K[j] = e_{j+k}
    so  Q[i] . K[j] = 1  iff  i = j + k  iff  j = i - k.
  * scores scaled by `temp` and softmaxed over keys -> ~all mass on key i-k.

Because the head's W_Q/W_K zero out the token block, the head provably *ignores*
token identity (matching the task: the shift is purely positional) while the
tokens are still genuinely present in the residual stream.

We also run a CAUSAL ABLATION: replace the shift matrix in W_K with the identity
(k -> 0). The dot product then peaks on the diagonal (query attends to itself),
so the mass on the shift-by-k target collapses to ~chance. This is the "knock out
the circuit and watch the behaviour break" evidence pass_2 lacked.

Everything runs in torch on CUDA.
"""

import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # guaranteed by the pipeline; no CPU fallback

# Canonical constants (mirror task.py).
SEQ_LEN = 32
BATCH = 8
VOCAB = 64
K_SWEEP = (1, 2, 3, 4, 8)   # one dedicated head per offset
N_HEADS = len(K_SWEEP)
D_HEAD = SEQ_LEN            # positional block width
D_MODEL = 2 * SEQ_LEN      # [token block | positional block]
TEMP = 30.0                # logit scale -> sharp softmax

# Seeded token embedding so the residual stream genuinely depends on token IDs.
_rng = np.random.default_rng(0)
_TOKEN_EMBED = _rng.standard_normal((VOCAB, SEQ_LEN)).astype(np.float32) * 0.5


def _shift_matrix(k: int) -> np.ndarray:
    """(L, L) matrix S with S[j, j+k] = 1 (for j+k < L), else 0.

    For a positional one-hot e_j (row vector), e_j @ S = e_{j+k}: it moves the
    active position forward by k, which is what turns the diagonal into the
    k-th sub-diagonal once we take Q[i] . K[j].
    """
    S = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float32)
    for j in range(SEQ_LEN - k):
        S[j, j + k] = 1.0
    return S


def _projections(shift: bool):
    """Build per-head W_Q, W_K of shape (H, d_model, d_head) on the GPU.

    W_Q reads the positional block with identity (Q[i] = e_i).
    W_K reads the positional block with the shift matrix S_k (K[j] = e_{j+k}),
    or — when ``shift=False`` (ablation) — with the identity (K[j] = e_j), which
    removes the offset and makes the head attend to itself.
    """
    Wq, Wk = [], []
    for k in K_SWEEP:
        wq = np.zeros((D_MODEL, D_HEAD), dtype=np.float32)
        wq[SEQ_LEN:, :] = np.eye(SEQ_LEN, dtype=np.float32)          # pick pos block
        wk = np.zeros((D_MODEL, D_HEAD), dtype=np.float32)
        wk[SEQ_LEN:, :] = _shift_matrix(k) if shift else np.eye(SEQ_LEN, dtype=np.float32)
        Wq.append(wq)
        Wk.append(wk)
    Wq = torch.as_tensor(np.stack(Wq), device=DEVICE)   # (H, d_model, d_head)
    Wk = torch.as_tensor(np.stack(Wk), device=DEVICE)
    return Wq, Wk


def make_model_fn(shift: bool = True):
    """Return a model_fn implementing the shift-by-k attention circuit on CUDA.

    Maps token IDs (B, L) -> attention (B, H, L, L) with all compute (embedding
    lookup, two projections, Q@K^T, softmax) done in torch on the GPU.
    ``shift=False`` yields the ablated (no-offset) circuit.
    """
    Wq, Wk = _projections(shift)
    tok_embed = torch.as_tensor(_TOKEN_EMBED, device=DEVICE)        # (V, L)
    pos_onehot = torch.eye(SEQ_LEN, device=DEVICE)                  # (L, L)

    def model_fn(input_ids: np.ndarray) -> np.ndarray:
        B, L = input_ids.shape
        assert L == SEQ_LEN, f"seq_len {L} != {SEQ_LEN}"
        ids = torch.as_tensor(input_ids.astype(np.int64), device=DEVICE)  # (B, L)

        # Residual stream X = [token block | positional block]  -> (B, L, d_model)
        tok = tok_embed[ids]                                   # (B, L, L)
        pos = pos_onehot.unsqueeze(0).expand(B, -1, -1)        # (B, L, L)
        X = torch.cat([tok, pos], dim=-1)                      # (B, L, 2L)

        # Per-head projections: (B,L,d_model) x (H,d_model,d_head)
        Q = torch.einsum("bld,hde->bhle", X, Wq)               # (B, H, L, d_head)
        K = torch.einsum("bld,hde->bhle", X, Wk)               # (B, H, L, d_head)

        # Honest attention: scores then softmax over keys.
        scores = torch.matmul(Q, K.transpose(-1, -2)) * TEMP   # (B, H, L, L)
        attn = torch.softmax(scores, dim=-1)
        return attn.detach().cpu().numpy().astype(np.float32)

    return model_fn


def _sweep_mass(payload):
    """{k: best_head_mass} from a payload's sweep."""
    return {s["k"]: s["best_head_mass"] for s in payload["sweep"]}


def _run():
    task = load_task(__file__)

    # --- The real shift-by-k circuit ---
    mech_fn = make_model_fn(shift=True)
    mech_payload = task.evaluate(mech_fn)
    mech_payload["model_name"] = "qk_shift_circuit (hand-built, real Q@K^T)"

    # --- Causal ablation: remove the shift matrix in W_K (k -> 0) ---
    abl_fn = make_model_fn(shift=False)
    abl_payload = task.evaluate(abl_fn)

    # --- Uniform / no-signal strawman from the task itself ---
    uni_fn = task.random_model_fn()
    uni_payload = task.evaluate(uni_fn)

    run_dir = results_dir(__file__)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save the per-head attention (first batch item) for the heatmap viz.
    batch = task.generate(0)
    attn_full = mech_fn(batch.input_ids)                  # (B, H, L, L)
    np.save(run_dir / "attn_heads.npy", attn_full[0])     # (H, L, L)

    mech_mass = _sweep_mass(mech_payload)
    abl_mass = _sweep_mass(abl_payload)
    uni_mass = _sweep_mass(uni_payload)

    summary = {
        "k_values": list(K_SWEEP),
        "head_for_k": {str(k): K_SWEEP.index(k) for k in K_SWEEP},
        "mechanism_mass": {str(k): mech_mass[k] for k in K_SWEEP},
        "ablation_mass": {str(k): abl_mass[k] for k in K_SWEEP},
        "uniform_mass": {str(k): uni_mass[k] for k in K_SWEEP},
        "mechanism_argmax_acc": {
            str(s["k"]): s["best_head_argmax_acc"] for s in mech_payload["sweep"]
        },
        "uniform_baseline": mech_payload["uniform_baseline"],
    }
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (run_dir / "payloads.json").open("w") as f:
        json.dump(
            {"mechanism": mech_payload, "ablation": abl_payload, "uniform": uni_payload},
            f, indent=2,
        )

    # Record the real circuit in the shared benchmark.
    record_benchmark(__file__, run_dir, mech_payload)

    print(f"Done -> {run_dir}")
    print(f"  mechanism mass per k : {mech_mass}")
    print(f"  ablation  mass per k : {abl_mass}")
    print(f"  uniform   mass per k : {uni_mass}")


if __name__ == "__main__":
    _run()
