"""attention_dot_product / pass_5

Hand-built scaled dot-product attention on CUDA, PLUS a causal ablation study.

The headline `model_fn` is the exact reference mechanism
`softmax(Q Kᵀ / √d_head) · V`, expressed as torch tensors on the GPU. That
alone reproduces the task's `gt_out` to machine precision and is what
`task.evaluate` scores into `benchmark.json`.

The *interesting* part is faithfulness: we re-run the SAME evaluation while
knocking out one component of the circuit at a time (the QKᵀ dot product, the
1/√d scale, the softmax). Each knock-out is a real GPU forward pass through a
single parameterised attention kernel, so the resulting fidelity collapse is a
causal demonstration that every piece of the dot-product mechanism is load
bearing — not a claim, a measured curve. Those ablation sweeps are saved next
to the benchmark for the app to plot.
"""

import json
from pathlib import Path

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

DEVICE = "cuda"  # guaranteed by the pipeline; no CPU fallback


# ---------------------------------------------------------------------------
# One parameterised attention kernel on the GPU. Turning a flag off "ablates"
# that component of the scaled-dot-product circuit.
# ---------------------------------------------------------------------------
def _attention_gpu(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    *,
    use_dot: bool = True,
    use_scale: bool = True,
    use_softmax: bool = True,
) -> np.ndarray:
    """Scaled dot-product attention with individually removable components.

    Q, K, V : (batch, n_heads, seq_len, d_head)  -> same-shape output.

    use_dot=False     -> no QKᵀ; every key gets the same logit (uniform mixing).
    use_scale=False   -> drop the 1/√d_head temperature.
    use_softmax=False -> linear attention: use raw (mean-normalised) logits as
                         weights instead of a softmax distribution.
    """
    Qt = torch.as_tensor(Q, dtype=torch.float32, device=DEVICE)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=DEVICE)
    Vt = torch.as_tensor(V, dtype=torch.float32, device=DEVICE)

    seq_len = Qt.shape[-2]
    d_head = Qt.shape[-1]

    if use_dot:
        scores = torch.einsum("bhsd,bhtd->bhst", Qt, Kt)  # (B,H,S,S)
    else:
        # No dot product: identical logit for every (query, key) pair.
        scores = torch.zeros(
            Qt.shape[:-1] + (seq_len,), dtype=torch.float32, device=DEVICE
        )

    if use_scale:
        scores = scores / (d_head ** 0.5)

    if use_softmax:
        weights = torch.softmax(scores, dim=-1)
    else:
        # Linear attention: normalise the raw logits to sum to 1 over keys so
        # the output stays on the same scale as V, but without the softmax
        # non-linearity that concentrates mass on the best-matching keys.
        weights = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    out = torch.einsum("bhst,bhtd->bhsd", weights, Vt)
    return out.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Headline model_fn: the full, exact mechanism. This is what gets benchmarked.
# ---------------------------------------------------------------------------
def model_fn(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    return _attention_gpu(Q, K, V, use_dot=True, use_scale=True, use_softmax=True)


# Attention weight matrices for the canonical condition (for the heatmap).
def _attn_weights(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    Qt = torch.as_tensor(Q, dtype=torch.float32, device=DEVICE)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=DEVICE)
    d_head = Qt.shape[-1]
    scores = torch.einsum("bhsd,bhtd->bhst", Qt, Kt) / (d_head ** 0.5)
    return torch.softmax(scores, dim=-1).detach().cpu().numpy()


# The ablation variants we re-evaluate through the SAME task.evaluate harness.
_ABLATIONS = {
    "full (softmax·QKᵀ/√d)": dict(use_dot=True, use_scale=True, use_softmax=True),
    "no_scale (drop 1/√d)": dict(use_dot=True, use_scale=False, use_softmax=True),
    "no_softmax (linear)": dict(use_dot=True, use_scale=True, use_softmax=False),
    "no_dot (uniform mix)": dict(use_dot=False, use_scale=True, use_softmax=True),
}


def _make_fn(**flags):
    def _fn(Q, K, V):
        return _attention_gpu(Q, K, V, **flags)

    return _fn


def main() -> None:
    run_dir = results_dir(__file__)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Headline benchmark: the full mechanism.
    payload = task.evaluate(model_fn)

    # 2) Causal ablation study: re-run the identical sweep for each knock-out.
    #    Every variant is a real GPU forward pass; we keep the same per-seq_len
    #    metrics that the benchmark uses so the curves are directly comparable.
    ablation = {}
    for name, flags in _ABLATIONS.items():
        sub = task.evaluate(_make_fn(**flags))
        ablation[name] = [
            {
                "seq_len": r["seq_len"],
                "mse": r["mse"],
                "cos_sim": r["cos_sim"],
                "rel_error": r["rel_error"],
                "baseline_mse": r["baseline_mse"],
            }
            for r in sub["sweep"]
        ]
    with open(run_dir / "ablation.json", "w") as f:
        json.dump(ablation, f, indent=2)

    # 3) Canonical attention heatmap data (seq_len = 32).
    canon = task.generate(seed=0)
    attn = _attn_weights(canon.Q, canon.K)  # (B, H, S, S)
    np.save(run_dir / "attn_weights.npy", attn)
    np.save(run_dir / "canon_pred.npy", model_fn(canon.Q, canon.K, canon.V))
    np.save(run_dir / "canon_gt.npy", canon.gt_out)

    # 4) Record the headline benchmark.
    record_benchmark(__file__, run_dir, payload)

    canon_rec = next(r for r in payload["sweep"] if r["seq_len"] == 32)
    print(f"Benchmark written to {run_dir}/benchmark.json")
    print(f"Canonical (seq_len=32) cos_sim={canon_rec['cos_sim']:.12f} "
          f"mse={canon_rec['mse']:.3e}")
    for name, recs in ablation.items():
        c = next(r for r in recs if r["seq_len"] == 32)
        print(f"  ablation {name:24s} cos_sim={c['cos_sim']:+.4f} "
              f"mse={c['mse']:.4e}")


if __name__ == "__main__":
    main()
