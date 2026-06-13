"""attention_quantile / pass_3 — hand-built scaled-dot-product attention.

Mechanism (a minimal delta from base_model.py's attention block):

    attn = softmax( GAIN * scale * (Q @ K^T) )    # single attention head, no MLP

This IS the base_model attention softmax with a temperature knob. The ONLY
per-condition signal the model receives is `scale`. Heavier-tail conditions
get a higher temperature (sharper softmax -> a few keys dominate -> large
quantile_ratio); lighter-tail conditions get a low temperature (flatter
softmax -> mass spread out -> small quantile_ratio). No parameters are
learned: GAIN is a single fixed constant standing in for the magnitude a
trained transformer's Q/K projections would learn (random unit vectors have
tiny dot products, so a real head would scale them up). The whole circuit is
hand-set.

We also run two CAUSAL ABLATIONS through the exact same evaluator to show the
mechanism is the one responsible for the tail structure:
  * no_temperature : freeze the per-condition scale to 1.0 -> pareto and
    exponential conditions become identical -> the pareto/exponential
    separation collapses to ~1.0.
  * linear (no exp): replace softmax with relu-normalisation -> the heavy tail
    flattens toward the uniform baseline.
"""

import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # GPU guaranteed by the pipeline; no CPU fallback.

task = load_task(__file__)
run_dir = results_dir(__file__)

# Fixed logit gain: stands in for the learned Q/K projection magnitude.
# Per-condition variation comes ONLY from `scale`; GAIN is constant everywhere.
GAIN = 6.0


def make_attn_fn(gain: float, use_temperature: bool = True, use_softmax: bool = True):
    """Build a model_fn(queries, keys, scale) -> [n_q, n_k] attention matrix."""

    def fn(queries: np.ndarray, keys: np.ndarray, scale: float) -> np.ndarray:
        qt = torch.as_tensor(queries, dtype=torch.float32, device=DEVICE)
        kt = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)
        temp = float(scale) if use_temperature else 1.0
        logits = (qt @ kt.T) * (gain * temp)            # [n_q, n_k]
        if use_softmax:
            attn = torch.softmax(logits, dim=1)
        else:
            # linear (no-exp) ablation: drop the exponential. Shift logits to be
            # non-negative per query (subtract the row min) and L1-normalise, so
            # weights are an AFFINE function of the logits instead of an
            # exponential one. Same ordering, but the tail-amplifying exp is gone,
            # flattening attention toward the uniform baseline.
            shifted = logits - logits.min(dim=1, keepdim=True).values + 1e-6
            attn = shifted / shifted.sum(dim=1, keepdim=True)
        return attn.detach().cpu().numpy().astype(np.float32)

    return fn


# The mechanism we are submitting.
model_fn = make_attn_fn(GAIN, use_temperature=True, use_softmax=True)

# Ablations (run through the identical task.evaluate path).
no_temp_fn = make_attn_fn(GAIN, use_temperature=False, use_softmax=True)
linear_fn = make_attn_fn(GAIN, use_temperature=True, use_softmax=False)


def summarize(payload: dict) -> dict:
    sweep = payload["sweep"]
    pareto = [r["quantile_ratio"] for r in sweep if r["tail_type"] == "pareto"]
    exp = [r["quantile_ratio"] for r in sweep if r["tail_type"] == "exponential"]
    canonical = next(r["quantile_ratio"] for r in sweep if r["condition_id"] == "pareto_0p5")
    exp_mean = float(np.mean(exp)) if exp else 0.0
    lift = float(np.mean(pareto) / exp_mean) if exp_mean > 0 else float("nan")
    return {
        "canonical_quantile_ratio": float(canonical),
        "pareto_mean": float(np.mean(pareto)),
        "exp_mean": exp_mean,
        "pareto_vs_exponential_lift": lift,
    }


# ---- evaluate the submitted mechanism (this is the benchmarked payload) ----
payload = task.evaluate(model_fn=model_fn)

# ---- evaluate ablations for the demo's causal comparison ----
ablations = {
    "full": summarize(payload),
    "no_temperature": summarize(task.evaluate(model_fn=no_temp_fn)),
    "linear_no_exp": summarize(task.evaluate(model_fn=linear_fn)),
    "uniform_baseline": {  # uniform attention: ratio is exactly 1.0 everywhere
        "canonical_quantile_ratio": 1.0,
        "pareto_mean": 1.0,
        "exp_mean": 1.0,
        "pareto_vs_exponential_lift": 1.0,
    },
}

# ---- a concrete attention example at the canonical condition (pareto_0p5) ----
batch = task.generate()
canon_idx = batch.condition_ids.index("pareto_0p5")
canon_scale = float(batch.scales[canon_idx])
attn_canon = model_fn(batch.queries, batch.keys, canon_scale)      # [n_q, n_k]
# Average sorted-descending attention profile + Lorenz (cumulative) curve.
sorted_desc = np.sort(attn_canon, axis=1)[:, ::-1].mean(axis=0)     # [n_k]
lorenz = np.cumsum(sorted_desc)                                     # rises to 1.0
n_keys = attn_canon.shape[1]
uniform_lorenz = np.cumsum(np.full(n_keys, 1.0 / n_keys))

# ---- persist artefacts for the app ----
artefact = {
    "config": payload["config"],
    "gain": GAIN,
    "canonical_scale": canon_scale,
    "sweep": [
        {
            "condition_id": r["condition_id"],
            "tail_type": r["tail_type"],
            "quantile_50": r["quantile_50"],
            "quantile_90": r["quantile_90"],
            "quantile_ratio": r["quantile_ratio"],
        }
        for r in payload["sweep"]
    ],
    "ablations": ablations,
    "sorted_desc_canonical": sorted_desc.astype(float).tolist(),
    "lorenz_canonical": lorenz.astype(float).tolist(),
    "lorenz_uniform": uniform_lorenz.astype(float).tolist(),
}
with open(f"{run_dir}/artefact.json", "w") as f:
    json.dump(artefact, f, indent=2)

print("full summary:        ", ablations["full"])
print("no_temperature abl.: ", ablations["no_temperature"])
print("linear (no-exp) abl.:", ablations["linear_no_exp"])

# ---- record the benchmark (this is what the leaderboard reads) ----
record_benchmark(__file__, run_dir, payload)
print("wrote benchmark + artefacts to", run_dir)
