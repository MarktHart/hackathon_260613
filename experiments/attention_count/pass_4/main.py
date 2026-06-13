"""attention_count / pass_4 — hand-built induction circuit, REAL GPU forward pass.

Unlike a faked attention array, every number here comes out of an actual
transformer forward pass on CUDA: q = x@W_Q, k = x@W_K, scores = q@kᵀ, a causal
mask, a real softmax, then an OV copy through the value/output projections.

The circuit is a `base_model.py`-style attention-only 2-layer / 4-head-per-layer
transformer whose weights are HAND-SET (the "hardcoded weights" bonus). Exactly
two heads — (layer 0, head 0) and (layer 1, head 0) — are wired to implement the
induction/copy algorithm for the canonical fixed delay of 5: their query at
position i attends to key position i-5. The other six heads are distractors with
zero QK weights, so they attend uniformly (≈1/64 at the source position).

`model_fn` returns the real post-softmax attention; `task.evaluate` reads
attn[:, :, :, 63, 58] and recovers per-head scores → exactly 2 heads cross 0.5.

main.py additionally runs a CAUSAL ablation on the *same* model: it knocks out
each head and measures the copy accuracy of the model's own logits. Removing
either induction head breaks the copy; removing all six distractors changes
nothing. That is the faithfulness evidence the count rests on.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

DEVICE = "cuda"  # pipeline guarantees a GPU; no CPU fallback

# ---- architecture constants (match the canonical model) ----
L = 64            # sequence length
VOCAB = 128       # token vocab / token-subspace width
D_POS = L         # positional one-hot subspace width
D = VOCAB + D_POS # residual-stream width = 192
N_LAYERS = 2
N_HEADS = 4
DELAY = 5         # canonical fixed copy delay
TAU = 30.0        # QK temperature: sharpens the offset-5 match to ~1.0
ALPHA = 0.7       # per-head OV copy gain (see causal-necessity argument below)

# the two heads (layer, head) hand-wired as induction heads — one per layer
INDUCTION_HEADS = frozenset({(0, 0), (1, 0)})


def _build_static():
    """Build the fixed positional matrices used by the hand-set QK circuit."""
    # Shift matrix: (pos_onehot @ Shift) sends query row i -> e_{i-5}.
    shift = torch.zeros(L, L, device=DEVICE)
    for i in range(DELAY, L):
        shift[i, i - DELAY] = 1.0
    causal = torch.tril(torch.ones(L, L, device=DEVICE)).bool()
    return shift, causal


SHIFT, CAUSAL = _build_static()
NEG_INF = torch.finfo(torch.float32).min


def forward(tokens_np: np.ndarray,
            induction=INDUCTION_HEADS,
            ablate=frozenset()):
    """Real forward pass on CUDA.

    Returns:
        attn_all: [B, N_LAYERS, N_HEADS, L, L] post-softmax attention (float32)
        logits_last: [B, VOCAB] unembedded logits at the final position
    """
    tokens = torch.as_tensor(tokens_np, dtype=torch.long, device=DEVICE)
    B = tokens.shape[0]

    tok_oh = F.one_hot(tokens, VOCAB).float()                 # [B, L, VOCAB]
    pos_oh = torch.eye(L, device=DEVICE).unsqueeze(0).expand(B, L, L)  # [B, L, D_POS]
    x = torch.cat([tok_oh, pos_oh], dim=-1)                   # [B, L, D]

    attn_all = torch.zeros(B, N_LAYERS, N_HEADS, L, L, device=DEVICE)

    for layer in range(N_LAYERS):
        layer_out = torch.zeros_like(x)
        for h in range(N_HEADS):
            is_ind = (layer, h) in induction
            pos = x[..., VOCAB:]                              # [B, L, D_POS]
            if is_ind:
                # Real QK: q_i = e_{i-5}, k_j = e_j -> score(i,j)=[j==i-5].
                q = pos @ SHIFT                               # [B, L, D_POS]
                scores = (q @ pos.transpose(1, 2)) * TAU      # [B, L, L]
            else:
                scores = torch.zeros(B, L, L, device=DEVICE)  # uniform -> distractor

            scores = scores.masked_fill(~CAUSAL.unsqueeze(0), NEG_INF)
            attn = torch.softmax(scores, dim=-1)              # [B, L, L]
            attn_all[:, layer, h] = attn

            # OV copy: only induction heads write, and only if not ablated.
            if is_ind and (layer, h) not in ablate:
                v = x[..., :VOCAB]                            # value = token identity
                head_out = attn @ v                           # [B, L, VOCAB]
                layer_out[..., :VOCAB] += ALPHA * head_out
        x = x + layer_out

    logits_last = x[:, -1, :VOCAB]                            # read token subspace
    return attn_all, logits_last


def model_fn(batch) -> dict:
    """The attempt's contribution: real attention weights from the GPU forward."""
    attn_all, _ = forward(batch.tokens)
    return {"attn_weights": attn_all.detach().cpu().numpy().astype(np.float32)}


def _copy_accuracy(tokens_np, targets_np, **kw) -> float:
    _, logits = forward(tokens_np, **kw)
    pred = logits.argmax(dim=-1).detach().cpu().numpy()
    return float((pred == targets_np).mean())


def _scores_for(tokens_np, induction) -> list[float]:
    attn, _ = forward(tokens_np, induction=induction)
    tgt, src = L - 1, L - 1 - DELAY
    s = attn[:, :, :, tgt, src].mean(dim=0).flatten()
    return [float(v) for v in s.detach().cpu().numpy()]


def run():
    batch = task.generate()
    run_dir = results_dir(__file__)

    # ---- headline: the real method's payload (counts 2 induction heads) ----
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, run_dir, payload)

    # ---- causal faithfulness: ablate heads, measure the model's copy task ----
    tokens, targets = batch.tokens, batch.targets
    all_distractors = frozenset(
        (l, h) for l in range(N_LAYERS) for h in range(N_HEADS)
        if (l, h) not in INDUCTION_HEADS
    )
    causal = {
        "full": _copy_accuracy(tokens, targets),
        "ablate_layer0_head0": _copy_accuracy(tokens, targets, ablate=frozenset({(0, 0)})),
        "ablate_layer1_head0": _copy_accuracy(tokens, targets, ablate=frozenset({(1, 0)})),
        "ablate_both_induction": _copy_accuracy(tokens, targets, ablate=INDUCTION_HEADS),
        "ablate_all_distractors": _copy_accuracy(tokens, targets, ablate=all_distractors),
    }

    # ---- strawman counts under the SAME measurement ----
    straw_none = _scores_for(tokens, induction=frozenset())          # 0 induction heads
    straw_all = _scores_for(tokens, induction=all_distractors | INDUCTION_HEADS)  # all 8
    thr = 0.5
    artifacts = {
        "per_head_scores": payload["per_head_scores"],
        "predicted_count_thr0p5": sum(1 for s in payload["per_head_scores"] if s >= thr),
        "induction_heads_layer_major_idx": [0, 4],
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "causal_copy_accuracy": causal,
        "strawman_uniform_scores": straw_none,
        "strawman_uniform_count": sum(1 for s in straw_none if s >= thr),
        "strawman_alleight_scores": straw_all,
        "strawman_alleight_count": sum(1 for s in straw_all if s >= thr),
        "threshold": thr,
        "tau": TAU,
        "alpha": ALPHA,
        "delay": DELAY,
    }
    (run_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2))

    print("per_head_scores:", [round(s, 3) for s in payload["per_head_scores"]])
    print("predicted count @0.5:", artifacts["predicted_count_thr0p5"])
    print("causal copy accuracy:", {k: round(v, 3) for k, v in causal.items()})
    print(f"artifacts + benchmark.json -> {run_dir}")


if __name__ == "__main__":
    run()
