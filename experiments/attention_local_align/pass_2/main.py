"""
attention_local_align / pass_2 — a REAL QK circuit, not a hand-written matrix.

The previous attempt (first_pass) wrote the ground-truth sub-diagonal attention
matrix directly into the output. The jury (correctly) flagged this as
tautological: no transformer, no QK, no positional structure — it just printed
the answer.

This attempt expresses predecessor attention as an actual single-head attention
circuit, a minimal delta from `experiments/base_model.py`:

    delta over base_model.py
    ------------------------
    * keep token embedding + a single self-attention head; DROP the MLP.
    * concatenate sinusoidal POSITION features onto the residual stream
      (d = d_tok + d_pos, token features in [:d_tok], positions in [d_tok:]).
    * hand-set W_Q to a block-diagonal *rotation by delta* acting on the
      positional sub-space; hand-set W_K to the identity on the positional
      sub-space. Both zero out the token sub-space.

The attention score between query t and key s is then

    score(t, s) = (R_delta p_t) . p_s = p_{t+delta} . p_s
                = sum_k cos( omega_k * (t + delta - s) )

a Dirichlet-style kernel that PEAKS sharply at s = t + delta. For the canonical
predecessor head we set delta = -1, so every query selects its immediate
predecessor through the QK dot-product — the selection is *computed*, not
written down. The same code, with delta in {-2,-1,0,+1,+2}, produces a head
aligned to any shift; sweeping delta against the data's ground-truth shift gives
a clean identity matrix (the diagonal experiment below).

Everything runs in torch on CUDA.
"""

import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

# ---- circuit hyper-parameters (a small attention head) ----
D_TOK = 32          # token sub-space width (carries content; ignored by this head)
K_FREQ = 32         # number of sinusoidal frequency pairs  ->  d_pos = 64
SCALE = 2.0         # attention temperature (sharpens the QK kernel)
VOCAB = 64


# ----------------------------------------------------------------------------
# positional encoding + rotation (the heart of the QK circuit)
# ----------------------------------------------------------------------------
def _freqs(k: int, device) -> torch.Tensor:
    # frequencies spread across (0, pi): makes the cos-kernel a sharp delta at 0
    idx = torch.arange(1, k + 1, device=device, dtype=torch.float32)
    return torch.pi * idx / (k + 1)


def pos_encoding(T: int, k: int, device) -> torch.Tensor:
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]   # (T,1)
    w = _freqs(k, device)[None, :]                                       # (1,k)
    ang = pos * w                                                        # (T,k)
    P = torch.zeros(T, 2 * k, device=device)
    P[:, 0::2] = torch.sin(ang)     # pair k -> dim 2k   = sin(omega_k t)
    P[:, 1::2] = torch.cos(ang)     # pair k -> dim 2k+1 = cos(omega_k t)
    return P                                                            # (T, 2k)


def rotation(delta: float, k: int, device) -> torch.Tensor:
    """Block-diagonal rotation R with R @ p_t = p_{t+delta}."""
    w = _freqs(k, device)
    c = torch.cos(w * delta)
    s = torch.sin(w * delta)
    R = torch.zeros(2 * k, 2 * k, device=device)
    for j in range(k):
        R[2 * j, 2 * j] = c[j]
        R[2 * j, 2 * j + 1] = s[j]
        R[2 * j + 1, 2 * j] = -s[j]
        R[2 * j + 1, 2 * j + 1] = c[j]
    return R                                                            # (2k, 2k)


def build_params(T: int, delta: float):
    P = pos_encoding(T, K_FREQ, DEVICE)            # (T, d_pos)
    d_pos = 2 * K_FREQ
    d = D_TOK + d_pos
    R = rotation(delta, K_FREQ, DEVICE)            # (d_pos, d_pos)
    Wq = torch.zeros(d_pos, d, device=DEVICE)
    Wq[:, D_TOK:] = R                              # rotate the positional sub-space
    Wk = torch.zeros(d_pos, d, device=DEVICE)
    Wk[:, D_TOK:] = torch.eye(d_pos, device=DEVICE)
    return P, Wq, Wk, d_pos


_TOK_TABLE = None


def _tok_table(d_tok: int) -> torch.Tensor:
    global _TOK_TABLE
    if _TOK_TABLE is None:
        g = torch.Generator(device=DEVICE).manual_seed(0)
        _TOK_TABLE = torch.randn(VOCAB, d_tok, generator=g, device=DEVICE)
    return _TOK_TABLE


# ----------------------------------------------------------------------------
# forward pass: a single attention head, returns (B, 1, T, T)
# ----------------------------------------------------------------------------
def forward_attn(
    input_ids: np.ndarray,
    delta: float,
    scale: float = SCALE,
    zero_tok: bool = False,
    zero_pos: bool = False,
    identity_rot: bool = False,
) -> torch.Tensor:
    input_ids = np.asarray(input_ids)
    B, T = input_ids.shape
    use_delta = 0.0 if identity_rot else float(delta)

    P, Wq, Wk, d_pos = build_params(T, use_delta)

    # token (content) features — random embedding indexed by (id mod VOCAB)
    ids_t = torch.as_tensor(input_ids.astype(np.int64), device=DEVICE) % VOCAB
    tok_feat = _tok_table(D_TOK)[ids_t]                     # (B,T,d_tok)
    if zero_tok:
        tok_feat = torch.zeros_like(tok_feat)

    pos_feat = P[None].expand(B, T, d_pos).clone()          # (B,T,d_pos)
    if zero_pos:
        pos_feat = torch.zeros_like(pos_feat)

    x = torch.cat([tok_feat, pos_feat], dim=-1)             # (B,T,d)
    q = x @ Wq.T                                            # (B,T,d_pos) = p_{t+delta}
    k = x @ Wk.T                                            # (B,T,d_pos) = p_s
    scores = scale * (q @ k.transpose(1, 2))               # (B,T,T)
    attn = torch.softmax(scores, dim=-1)                   # softmax over keys
    return attn[:, None, :, :]                             # (B,1,T,T)


def make_model_fn(delta: float):
    def _fn(input_ids: np.ndarray) -> np.ndarray:
        return (
            forward_attn(input_ids, delta=delta, scale=SCALE)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    return _fn


# canonical predecessor head used for the headline benchmark
model_fn = make_model_fn(delta=-1.0)


# ----------------------------------------------------------------------------
# analyses (baseline, faithfulness, operating range) saved as artefacts
# ----------------------------------------------------------------------------
def _canonical_align(attn_b1tt: np.ndarray, target_indices: np.ndarray) -> float:
    a = attn_b1tt[:, 0]                                     # (B,T,T)
    valid = target_indices != -1
    safe = np.where(valid, target_indices, 0)
    gathered = np.take_along_axis(a, safe[:, :, None], axis=2)[:, :, 0]
    return float(gathered[valid].mean())


def run_analyses(task) -> dict:
    out = {}

    # --- the canonical predecessor head, its full sweep + a random strawman ---
    payload = task.evaluate(model_fn)
    out["model_sweep"] = [
        {
            "shift": s["shift"],
            "align": s["mean_max_attn_to_target"],
            "entropy": s["mean_entropy"],
            "peak": s["frac_peak_on_target"],
        }
        for s in payload["sweep"]
    ]
    rand_payload = task.evaluate(task.random_model_fn())
    out["random_sweep"] = [
        {"shift": s["shift"], "align": s["mean_max_attn_to_target"]}
        for s in rand_payload["sweep"]
    ]

    # --- offset x data-shift matrix: model rotated by delta aligns to shift=delta ---
    offsets = [-2, -1, 0, 1, 2]
    matrix = []
    for md in offsets:
        pl = task.evaluate(make_model_fn(float(md)))
        matrix.append([s["mean_max_attn_to_target"] for s in pl["sweep"]])
    out["offsets"] = offsets
    out["data_shifts"] = [s["shift"] for s in payload["sweep"]]
    out["offset_shift_matrix"] = matrix

    # --- faithfulness: ablate parts of the canonical (shift=-1) circuit ---
    canon = task.generate(seed=0)
    ids = canon.input_ids
    tgt = canon.target_indices
    T = ids.shape[1]
    rng = np.random.default_rng(7)
    ids_shuf = np.stack([rng.permutation(row) for row in ids])

    def ca(**kw):
        return _canonical_align(
            forward_attn(ids if "ids" not in kw else kw.pop("ids"), -1.0, SCALE, **kw)
            .detach().cpu().numpy(),
            tgt,
        )

    out["ablations"] = {
        "full": ca(),
        "zero_tokens": ca(zero_tok=True),
        "shuffle_tokens": _canonical_align(
            forward_attn(ids_shuf, -1.0, SCALE).detach().cpu().numpy(), tgt
        ),
        "zero_positions": ca(zero_pos=True),
        "identity_rotation": ca(identity_rot=True),
        "uniform_baseline": 1.0 / (T - 1),
        "random_strawman": out["random_sweep"][1]["align"],  # shift=-1 entry
    }

    # --- operating range across sequence length (2+ orders of magnitude) ---
    curve = []
    rng2 = np.random.default_rng(0)
    for Tn in [8, 16, 32, 64, 128, 256, 512]:
        idn = rng2.integers(0, 2**31 - 1, size=(64, Tn)).astype(np.int32)
        a = forward_attn(idn, -1.0, SCALE)[:, 0].detach().cpu().numpy()  # (B,T,T)
        idx = np.arange(1, Tn)
        align = float(a[:, idx, idx - 1].mean())                        # attn on t-1
        curve.append({"T": Tn, "align": align})
    out["seqlen_curve"] = curve

    # --- one example attention matrix for a heatmap (canonical, first seq) ---
    ex = forward_attn(ids[:1], -1.0, SCALE)[0, 0].detach().cpu().numpy()  # (T,T)
    out["example_attn"] = ex.tolist()

    return payload, out


def main():
    task = load_task(__file__)

    payload, analysis = run_analyses(task)

    run_dir = results_dir(__file__)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    record_benchmark(__file__, run_dir, payload)

    print(f"Done. Results in {run_dir}")
    print("Canonical sweep (predecessor head, delta=-1):")
    for s in payload["sweep"]:
        print(
            f"  shift={s['shift']:+d}  align={s['mean_max_attn_to_target']:.4f}  "
            f"entropy={s['mean_entropy']:.4f}  peak={s['frac_peak_on_target']:.4f}"
        )
    print("Ablations (canonical alignment):")
    for k, v in analysis["ablations"].items():
        print(f"  {k:18s}: {v:.4f}")


if __name__ == "__main__":
    main()
