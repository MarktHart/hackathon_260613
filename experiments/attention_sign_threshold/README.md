# Attention Sign Threshold

## Question

How sharply does an attention head (or an interpretation method's prediction of it) transition from attending to anti-attending as the query–key dot product crosses zero? In other words: does attention implement a clean sign detector at dot-product = 0, or is the transition gradual and context-dependent?

## Setup

Fully synthetic. No trained model, no dataset. We generate teams of query/key pairs whose cosine similarity sweeps from −1 to +1. The **canonical measurement condition** is a 21-point linear sweep in cosine similarity:  
`cos_sim ∈ {−1.0, −0.9, −0.8, …, 0.0, …, 0.9, 1.0}`.

Each cosine value is realised by 100 random query–key pairs (same seed every run) to average out directional noise. Queries and keys are unit vectors in ℝ⁶⁴.

The **model function** supplied by an attempt has signature:

```python
def model_fn(queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """
    queries: (n_pairs, d_model) float32
    keys:    (n_pairs, d_model) float32
    returns: (n_pairs,) float32  -- attention weight (or logit) for each pair
    """
```

The function must be deterministic and pure (no RNG, no I/O). It may be a real model's attention head extracted via hooks, a learned probe, a hand-coded heuristic, etc. The benchmark only sees the scalar output per pair.

## Canonical Measurement Condition

- **Sweep axis**: `cosine_similarity` from −1.0 to +1.0 in steps of 0.1 (21 points).
- **Pairs per point**: 100 (fixed seed 42).
- **Dimension**: 64.
- **Aggregation**: mean attention weight per cosine bin.
- **Normalisation**: the evaluator maps each pair's raw score to an attention weight in `[0, 1]` *per pair*. If the model already returns weights in `[0, 1]`, they are used as-is; otherwise the score is treated as a logit and squashed with a `sigmoid`. We deliberately **do not** softmax within a bin: every pair in a bin shares the same cosine, so a per-bin softmax would force the bin mean to exactly `1/pairs_per_bin` for *every* model and erase the signal the sweep measures. The per-pair sigmoid keeps each bin mean free to vary with cosine and makes different attempts comparable on a common `[0, 1]` scale.

## Payload Contract

`task.evaluate(model_fn)` returns a dict with exactly these keys:

```python
{
    "version": 1,                              # payload version, not benchmark version
    "config": {
        "d_model": 64,
        "pairs_per_bin": 100,
        "cosine_sweep": [-1.0, -0.9, ..., 1.0],  # 21 floats
        "seed": 42,
        "normalisation": "sigmoid_per_pair"      # fixed by evaluator
    },
    "sweep": [
        {"cosine": -1.0, "mean_attention": 0.0200, "std_attention": 0.0030},
        {"cosine": -0.9, "mean_attention": 0.0260, "std_attention": 0.0041},
        ...
        {"cosine": 1.0, "mean_attention": 0.9700, "std_attention": 0.0056}
    ],
    "model_info": {
        "name": "attempt-specific identifier",
        "type": "head_hook | probe | heuristic | ...",
        "notes": "free text"
    }
}
```

- `sweep` is a list of 21 records, one per cosine value in ascending order.
- `mean_attention` and `std_attention` are computed over the 100 pairs in that bin *after* the evaluator maps each pair's score to an attention weight in `[0, 1]` (per-pair sigmoid, or pass-through if already in `[0, 1]`). The example values above are illustrative, not literal.
- `model_info` is populated by the attempt's `main.py` and passed through unchanged; `score()` does not read it.

## Metrics

`benchmark.score(payload)` returns a flat dict:

| Metric | Formula / Description | Better |
|--------|----------------------|--------|
| `version` | `benchmark.VERSION` (currently 1) | — |
| `sign_sharpness_canonical` | Slope of a logistic fit `σ(α·(cos − θ))` to the sweep at the canonical sweep. α is the sharpness (steepness). Larger α = sharper threshold. | Larger |
| `sign_threshold_canonical` | θ from the same logistic fit: the cosine value where attention = 0.5. Ideally 0.0. | Closer to 0 |
| `sign_sharpness_cos_<val>` | Per-slice: local finite-difference slope `(μ_{i+1} − μ_{i−1}) / (cos_{i+1} − cos_{i−1})` at each **interior** cosine. Only the 19 interior points have both neighbours, so the endpoints ±1.0 are omitted. Named `cos_m0p9`, `cos_m0p8`, … `cos_0p0`, … `cos_0p9` (dots → `p`, leading minus → `m`). | Larger near 0; sign indicates direction |
| `linear_baseline_sharpness_canonical` | Same logistic-fit α applied to the **linear baseline** `attention = max(0, cos)` (already in [0,1], evaluated under the same fit). This is the no-mechanism reference. | Reference |
| `lift_over_linear_sharpness` | `sign_sharpness_canonical / linear_baseline_sharpness_canonical`. Ratio > 1 means the method detects the sign transition more sharply than a linear ramp. If the baseline α collapses to ~0 the ratio is capped at `1e9` to stay JSON-serialisable. | Larger |

All metrics are floats. `sign_sharpness_*` and `linear_baseline_sharpness_*` are ≥ 0. `sign_threshold_canonical` ∈ [−1, 1]. `lift_over_linear_sharpness` ∈ [0, ∞).

## Bump Procedure

Bump `benchmark.VERSION` (and the payload `version` if its keys change) when:
- The sweep axis, range, or granularity changes.
- The normalisation rule changes.
- Any metric formula changes.
- Payload keys are added/removed/retyped.

Adding a new per-slice metric or a new baseline does **not** require a version bump.

## Implementation Notes for Attempt Authors

1. Import `generate` and `evaluate` from `task.py`; do not re-implement the data.
2. Your `main.py` should call `payload = evaluate(your_model_fn)` and write the returned dict verbatim to `benchmark.json`.
3. The provided `random_model_fn()` returns all zeros — a no-mechanism reference. Use it for a pipeline smoke test:
   ```python
   from task import evaluate, random_model_fn
   payload = evaluate(random_model_fn())
   # The sweep is flat, so sign_sharpness_canonical ≈ 0 and the attempt is
   # (correctly) flagged as having no sign-detection mechanism.
   ```
4. If your method produces logits, the evaluator squashes them per pair with a sigmoid. If it already produces weights in `[0, 1]`, pass them as-is — the evaluator detects the range and skips the sigmoid.
5. `model_info` in the payload is free-form; put whatever helps you debug.