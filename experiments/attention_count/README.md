# attention_count

## Question

Given a transformer model trained on a synthetic copying task where exactly **K attention heads** implement the induction-head algorithm (copying from previous occurrences), can an interpretability method correctly **count** how many heads are performing that algorithm?

The method should output a scalar estimate of the number of induction heads. We measure accuracy of that count against the ground-truth K.

## Setup

- **Synthetic generator**: A 2-layer, 4-head-per-layer attention-only transformer (total 8 heads) trained on a *copies* task: sequences of random tokens where the target is the token that appeared `delay` positions earlier. After training, exactly **2 heads** (one per layer) reliably implement the induction algorithm; the other 6 heads are distracted/noise.
- **Canonical model**: A fixed checkpoint (`canonical_model.pt`) shipped with the goal. All attempts evaluate the *same* weights — no retraining.
- **Canonical dataset**: 256 sequences of length 64, vocab size 128, fixed `delay = 5` (every sequence's copy source is 5 positions before the target). Same seed every run. The induction score for each head is read off at `source_pos = target_pos - 5`, so the measurement offset is tied to this single canonical delay.
- **Measurement condition**: Single forward pass per sequence; collect per-head attention patterns (query-key dot products) at the target position. No patching, no ablation — just static analysis of the attention weights.

## Model function signature

```python
def model_fn(batch: Batch) -> dict[str, np.ndarray]:
    """
    Args:
        batch: Batch with fields
            tokens:  int32[B, L]   input token ids
            targets: int32[B]      target token at the copy position
    Returns:
        dict with key:
            attn_weights: float32[B, n_layers, n_heads, L, L]
                Attention weights (post-softmax) for every head, every layer.
                Layer 0 first, then layer 1. Heads in model order.
    """
```

The attempt's `main.py` implements this function by loading the canonical checkpoint and running a forward pass with hooks to capture the attention weights.

## Payload contract

`task.evaluate(model_fn)` returns a dict with exactly these keys:

```python
{
    "version": 1,                              # payload version
    "n_layers": 2,                             # fixed
    "n_heads": 4,                              # per layer
    "ground_truth_induction_heads": 2,         # known from training dynamics
    "per_head_scores": list[float],            # length 8, one score per head in layer-major order
    "threshold_sweep": list[dict],             # length 21, thresholds 0.0 .. 1.0 step 0.05
        each dict: {"threshold": float, "predicted_count": int}
}
```

- `per_head_scores[i]` ∈ [0, 1] is the method's confidence that head `i` is an induction head. Higher = more confident.
- `threshold_sweep` is produced by thresholding `per_head_scores` at each threshold and counting heads ≥ threshold.

## Metrics

Returned by `benchmark.score(payload)`:

| metric | formula | direction |
|--------|---------|-----------|
| `count_accuracy_canonical` | `1.0 - abs(pred_count - 2) / 2` at threshold 0.5 | bigger better |
| `count_accuracy_thr_<t>` | same at each threshold `t` (0p00, 0p05, …, 1p00) | bigger better |
| `auc_count` | Area under the `count_accuracy_thr` curve (trapezoid) | bigger better |
| `baseline_accuracy_canonical` | Accuracy of always guessing 4 (midpoint) | reference |
| `lift_over_baseline` | `count_accuracy_canonical - baseline_accuracy_canonical` | bigger better |
| `version` | payload version | — |

All accuracies in [0, 1]. `count_accuracy_canonical` is the **headline summary**.

## Bump procedure

- Bump `VERSION` in `benchmark.py` and `version` in payload when:
  - Ground-truth K changes (model retrained).
  - Threshold grid changes.
  - Metric formulas change.
- Add new metrics without bumping.
- Update `README.md` benchmark-contract section in the same commit.