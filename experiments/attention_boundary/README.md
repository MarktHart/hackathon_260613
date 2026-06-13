# Attention Boundary Detection

## Question

Can an interpretability method reliably identify which attention heads function as **boundary detectors** — heads that consistently attend to structural boundaries (e.g., indentation changes, bracket pairs, sentence boundaries) — and localize the boundary positions themselves?

## Setup

**Synthetic generator only.** A minimal 4-head attention model with fully known ground truth:
- **Head 0**: Local backward (attends to previous token)
- **Head 1**: Local forward (attends to next token)
- **Head 2**: Boundary detector (attends uniformly to all boundary tokens)
- **Head 3**: Content detector (attends uniformly to all non-boundary tokens)

Sequences are length 64 over vocabulary `{0: PAD, 1: CONTENT, 2: BOUNDARY}`. Boundaries are explicit `BOUNDARY` tokens placed at regular intervals (every 16 positions by default). The synthetic model computes attention patterns analytically — no Torch, no GPU, deterministic.

**Canonical measurement condition**
- `seq_len = 64`
- `num_sequences = 100`
- `boundary_spacing = 16` (boundaries at positions 16, 32, 48)
- `num_heads = 4`
- `true_boundary_heads = [2]`
- `seed = 42` (fixed; `generate` ignores the seed argument but accepts it for API compatibility)

## Model-function contract

Every attempt must provide a `model_fn` with this exact signature:

```python
def model_fn(tokens: list[list[int]]) -> dict:
    """
    Analyse the synthetic model on `tokens` and return boundary/head predictions.

    Args:
        tokens: List of N sequences, each a list of L token IDs (0, 1, or 2).

    Returns:
        dict with three required keys:
        - "pred_boundaries": list[list[int]] — predicted boundary positions per sequence.
        - "head_boundary_scores": list[float] — length 4, higher = more boundary-like.
        - "predicted_boundary_heads": list[int] — indices of heads classified as boundary detectors.
    """
```

The attempt’s `main.py` should import `synthetic_model` from `task.py` (or re-implement it identically) and close it over inside `model_fn`. The framework’s smoke test calls `task.evaluate(task.random_model_fn())`; `random_model_fn` returns a function matching the signature above that emits random predictions.

## Payload contract

`task.evaluate` returns a `dict` with this exact structure (Python types shown):

```python
{
    "version": 1,                                    # int, matches benchmark.VERSION
    "config": {                                      # dict, frozen run configuration
        "seq_len": 64,
        "num_sequences": 100,
        "boundary_spacing": 16,
        "num_heads": 4,
        "true_boundary_heads": [2],
        "canonical_seed": 42,
        "boundary_type": "explicit_token"
    },
    "sweep": [                                       # list[dict], one entry per condition
        {
            "condition": "canonical",                # str, condition identifier
            "boundary_f1": 0.0,                      # float, harmonic mean of precision/recall
            "boundary_precision": 0.0,               # float
            "boundary_recall": 0.0,                  # float
            "head_detection_accuracy": 0.0,          # float, fraction of heads correctly classified
            "head_precision": 0.0,                   # float, TP / (TP + FP) for head classification
            "head_recall": 0.0,                      # float, TP / (TP + FN) for head classification
            "num_sequences": 100                     # int
        }
        # Future conditions (e.g., variable spacing) append here — same keys.
    ]
}
```

All floats are in `[0, 1]`. `sweep` is a list to allow future multi-condition benchmarks without contract changes.

## Metrics

`benchmark.score(payload)` returns a flat `dict[str, float | int]`:

| Metric | Formula | Direction | Notes |
|--------|---------|-----------|-------|
| `version` | `payload["version"]` | — | First key, dashboard filter |
| `boundary_f1_canonical` | `sweep[0]["boundary_f1"]` | **Bigger = better** | Headline summary metric |
| `boundary_precision_canonical` | `sweep[0]["boundary_precision"]` | Bigger = better | Per-slice |
| `boundary_recall_canonical` | `sweep[0]["boundary_recall"]` | Bigger = better | Per-slice |
| `head_accuracy_canonical` | `sweep[0]["head_detection_accuracy"]` | Bigger = better | Per-slice |
| `head_precision_canonical` | `sweep[0]["head_precision"]` | Bigger = better | Per-slice |
| `head_recall_canonical` | `sweep[0]["head_recall"]` | Bigger = better | Per-slice |
| `linear_baseline_boundary_f1_canonical` | Baseline F1 (random boundary guess) | Bigger = better | Reference |
| `linear_baseline_head_accuracy_canonical` | Baseline head accuracy (random head guess) | Bigger = better | Reference |
| `lift_over_baseline_boundary_f1_canonical` | `boundary_f1_canonical - linear_baseline_boundary_f1_canonical` | Bigger = better | Improvement over strawman |
| `lift_over_baseline_head_accuracy_canonical` | `head_accuracy_canonical - linear_baseline_head_accuracy_canonical` | Bigger = better | Improvement over strawman |

**Baseline computation** (deterministic, inside `benchmark.score`):
- Boundary baseline: Predict the empirical boundary rate uniformly at random per position. Expected F1 = `2 * p * r / (p + r)` where `p = r = (num_boundaries / seq_len) = 3/64 ≈ 0.0469`.
- Head baseline: Randomly pick `k` heads as boundary detectors where `k ~ Binomial(4, 0.5)`. Expected accuracy = 0.5.

**`is_obviously_broken`** returns `True` if any metric is NaN/Inf, or if `boundary_f1_canonical <= 1.5 * linear_baseline_boundary_f1_canonical` (i.e., method fails to beat a trivial random-guess baseline by a meaningful margin).

## Bump procedure

Bump `VERSION` in `benchmark.py` and update this README when:
- Any metric formula changes.
- Payload keys are added, removed, or retyped.
- Canonical condition parameters change (`seq_len`, `boundary_spacing`, etc.).
- Do **not** bump for adding new metrics or optional payload keys with defaults.