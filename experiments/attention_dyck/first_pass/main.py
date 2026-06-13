"""Hand-built Dyck stack-matching attention circuit.

Attempt type: hand_built (interp circuit expressed as a single attention layer
with two heads). We do NOT train. Instead we construct the post-softmax
attention weights a *trained* depth-tracking head would have to implement, and
hand the result to the goal's canonical evaluator.

Mechanism (delta from base_model.py): the model is one attention layer
(`base_model.Attention`) whose QK score for a closing-bracket query is the
content/position score
        score(i, j) = -ALPHA * (depth_j - (depth_i + 1))**2 + BETA * j
restricted to *opening* keys j < i. `depth_t = cumsum(+1 for '(', -1 for ')')`
is exactly the prefix nesting depth that RoPE/positional + a running-sum head
can expose. The matching open for a close at running depth `d_i` is the unique
most-recent open with running depth `d_i + 1`; the quadratic term selects that
depth band and the small +BETA*j term breaks ties toward recency (the correct
sibling). Head 0 implements this matcher. Head 1 instead spreads attention over
all prior opens with weight exp(GAMMA * depth_j) — a monotone depth code — so
its attention-to-opens correlates with nesting depth. Non-closing queries (and
the BOS/pad rows) park their mass on BOS, keeping the diagonal fraction ~0.

GPU: all of the score construction + softmax runs in torch on CUDA.
"""

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU; no CPU fallback

task = load_task(__file__)

OPEN = task.VOCAB["OPEN"]   # 1
CLOSE = task.VOCAB["CLOSE"]  # 2
BOS = task.VOCAB["BOS"]      # 3

# Circuit constants. ALPHA must dominate the recency term over the whole length
# (BETA * seq_len) so an exact depth match always beats a closer-but-wrong-depth
# open: ALPHA=200 >> BETA*64=64.
ALPHA = 200.0
BETA = 1.0
GAMMA = 1.2


def _build_attn(input_ids: np.ndarray, attention_mask: np.ndarray) -> torch.Tensor:
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=DEVICE)
    mask = torch.as_tensor(attention_mask, device=DEVICE).bool()
    B, S = ids.shape

    is_open = ids == OPEN
    is_close = ids == CLOSE
    delta = is_open.float() - is_close.float()
    running = torch.cumsum(delta, dim=1)  # (B, S) prefix nesting depth

    idx = torch.arange(S, device=DEVICE, dtype=torch.float32)
    qi = idx.view(1, S, 1)
    kj = idx.view(1, 1, S)
    causal = kj < qi                              # (1, S, S)
    open_key = is_open.view(B, 1, S)
    valid_key = mask.view(B, 1, S)
    key_ok = causal & open_key & valid_key        # (B, S, S): valid open keys before query

    run_i = running.view(B, S, 1)
    run_j = running.view(B, 1, S)
    target = run_i + 1.0

    neg = torch.finfo(torch.float32).min / 4.0

    # Head 0: depth-band matcher with recency tiebreak.
    score0 = -ALPHA * (run_j - target) ** 2 + BETA * kj
    score0 = torch.where(key_ok, score0, torch.full_like(score0, neg))
    attn0 = torch.softmax(score0, dim=-1)

    # Head 1: monotone depth code over all prior opens.
    score1 = (GAMMA * run_j).expand(B, S, S)
    score1 = torch.where(key_ok, score1, torch.full_like(score1, neg))
    attn1 = torch.softmax(score1, dim=-1)

    # Rows that are not closing brackets (or have no prior open) park on BOS (pos 0).
    has_key = key_ok.any(dim=-1)                  # (B, S)
    use = (is_close & has_key & mask).view(B, S, 1)
    bos = torch.zeros(B, S, S, device=DEVICE)
    bos[:, :, 0] = 1.0

    attn0 = torch.where(use, torch.nan_to_num(attn0), bos)
    attn1 = torch.where(use, torch.nan_to_num(attn1), bos)

    return torch.stack([attn0, attn1], dim=1)     # (B, 2, S, S)


def model_fn(input_ids: np.ndarray, attention_mask: np.ndarray) -> dict:
    attn = _build_attn(input_ids, attention_mask)
    return {"attn_weights": attn.detach().float().cpu().numpy()}


def main():
    run_dir = results_dir(__file__)

    payload = task.evaluate(model_fn)
    record_benchmark(__file__, run_dir, payload)

    # Save a small slice of the canonical batch for the Demo tab.
    batch = task.generate(task.CANONICAL_SEED)
    attn = model_fn(batch.input_ids, batch.attention_mask)["attn_weights"]
    lengths = batch.attention_mask.sum(axis=1)
    # Pick interesting examples: longest / deepest first.
    order = np.argsort(-lengths)
    keep = order[:16]
    np.savez_compressed(
        run_dir / "viz.npz",
        input_ids=batch.input_ids[keep],
        attn=attn[keep].astype(np.float32),
        matching_open_pos=batch.matching_open_pos[keep],
        open_depth=batch.open_depth[keep],
        labels=batch.labels[keep],
        lengths=lengths[keep],
        example_ids=keep,
    )

    agg = payload["aggregated"]
    print("best_matching_accuracy :", round(agg["best_matching_accuracy"], 4))
    print("best_depth_corr        :", round(agg["best_depth_corr"], 4))
    print("linear_baseline        :", round(agg["linear_baseline_matching"], 4))
    print("per_head               :")
    for ph in payload["per_head"]:
        print(
            f"  head {ph['head']}: match={ph['matching_accuracy']:.3f} "
            f"depth_corr={ph['depth_corr']:.3f} diag={ph['diag_frac']:.4f}"
        )
    print("results ->", run_dir)


if __name__ == "__main__":
    main()
