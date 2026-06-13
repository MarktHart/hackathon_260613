"""attention_dtw / first_pass — hand-built content-matching alignment head.

Hypothesis
----------
A time-warped key sequence `keys[n] = queries[align[n]] + eps` can be aligned
back to its source by *content* alone: for each key position the attention
should point at the query whose feature vector it most resembles. That is a
DTW-style alignment circuit, and because it ignores absolute position it tracks
the warp instead of running down the diagonal.

This is a HAND-BUILT attempt: no training. The model function is base_model.py's
self-attention with the QK dot-score swapped for a negative-squared-L2 distance
kernel and RoPE removed (position is exactly what we must NOT use). We expose
three heads so the demo can contrast the mechanism against strawmen:

    head 0  content_l2   : softmax(-||k_n - q_m||^2 / T)   <- the alignment circuit
    head 1  content_dot  : softmax( (k_n . q_m) / T )      <- base_model-style score
    head 2  diagonal     : softmax(-(pos_n - pos_m)^2 / T) <- position-only strawman

`evaluate` keeps the best head per example (content_l2 wins), while the diagonal
head is the no-mechanism baseline the README's metric also measures.

All real compute runs in torch on CUDA.
"""

from __future__ import annotations

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # guaranteed by the pipeline; no CPU fallback

# Head sharpness (temperature). Feature separation is huge (noise std 0.1 over
# D=8 vs. unit-variance queries), so a small temperature makes the argmax crisp.
T_CONTENT = 0.1
T_DIAG = 0.5

task = load_task(__file__)


def make_model_fn():
    """Return the hand-built ModelFn: (queries (M,D), keys (N,D)) -> (H, N, M)."""

    def model_fn(queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
        qt = torch.as_tensor(queries, dtype=torch.float32, device=DEVICE)  # (M, D)
        kt = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)     # (N, D)
        M = qt.shape[0]
        N = kt.shape[0]

        # --- head 0: content via negative squared L2 distance ---
        # ||k_n - q_m||^2  ->  (N, M)
        d2 = torch.cdist(kt, qt, p=2) ** 2
        content_l2 = torch.softmax(-d2 / T_CONTENT, dim=1)

        # --- head 1: content via raw dot product (base_model QK score) ---
        dot = kt @ qt.t()  # (N, M)
        content_dot = torch.softmax(dot / T_CONTENT, dim=1)

        # --- head 2: diagonal positional strawman (no content used) ---
        n_idx = torch.arange(N, device=DEVICE, dtype=torch.float32).unsqueeze(1)  # (N,1)
        m_idx = torch.arange(M, device=DEVICE, dtype=torch.float32).unsqueeze(0)  # (1,M)
        diag_target = n_idx * (M - 1) / (N - 1)
        diag = torch.softmax(-((diag_target - m_idx) ** 2) / T_DIAG, dim=1)

        attn = torch.stack([content_l2, content_dot, diag], dim=0)  # (3, N, M)
        return attn.detach().cpu().numpy()

    return model_fn


def _save_demo_artifacts(run_dir, model_fn):
    """Save one representative example per warp for the Demo tab."""
    batch = task.generate(0)
    head_names = ["content_l2", "content_dot", "diagonal"]
    per_warp = {}
    for warp in batch.warp_sweep:
        ex = batch.examples[warp][0]
        attn = model_fn(ex.queries, ex.keys)  # (H, N, M)
        preds = np.argmax(attn, axis=2)       # (H, N)
        per_warp[f"{warp:g}"] = {
            "attn": attn.astype(np.float32),
            "align": ex.align.astype(np.int64),
            "preds": preds.astype(np.int64),
        }
    np.savez_compressed(
        run_dir / "demo_examples.npz",
        head_names=np.array(head_names),
        warps=np.array([float(w) for w in batch.warp_sweep]),
        **{f"{k}__{field}": v[field]
           for k, v in per_warp.items() for field in ("attn", "align", "preds")},
    )


def main():
    model_fn = make_model_fn()
    payload = task.evaluate(model_fn)

    run_dir = results_dir(__file__)
    _save_demo_artifacts(run_dir, model_fn)
    record_benchmark(__file__, run_dir, payload)

    print("num_heads:", payload["num_heads"])
    for s, b in zip(payload["sweep"], payload["baseline"]):
        print(
            f"  warp={s['warp']:.2f}  best_head={s['best_head_overlap']:.3f}"
            f"  mean_head={s['mean_head_overlap']:.3f}"
            f"  mono={s['monotonicity']:.3f}"
            f"  diag_base={b['diagonal_overlap']:.3f}"
        )
    print("artifacts in:", run_dir)


if __name__ == "__main__":
    main()
