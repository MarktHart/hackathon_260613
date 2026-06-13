# attention_xor

## Question

Can a model compute the **XOR** of two binary features `A` and `B` — a function
that is **not** linearly separable in the input features — from a tokenised
encoding, and do so reliably enough to beat the *best possible linear probe*
across a sweep of feature marginals? A linear classifier over the input bits
tops out well below 100% accuracy on XOR (it must give up on at least one of the
four input cells); a genuine non-linear mechanism captures the gap the linear
floor leaves open.

## Setup

**Synthetic generator.** No trained model is required by the task itself. For
each marginal `p` in the sweep, two independent Bernoulli(`p`) draws produce the
binary features `A` and `B`; the label is `A XOR B`. Each example is encoded as
a length-4 integer token sequence

```
[CLS, A_tok, B_tok, SEP]
```

with the token vocabulary

| id | token | meaning            |
|----|-------|--------------------|
| 0  | CLS   | sequence start     |
| 1  | A0    | feature `A = 0`    |
| 2  | A1    | feature `A = 1`    |
| 3  | B0    | feature `B = 0`    |
| 4  | B1    | feature `B = 1`    |
| 5  | SEP   | sequence end       |

So `A_tok = A + 1 ∈ {1, 2}` and `B_tok = B + 3 ∈ {3, 4}`. The label for each row
is `A XOR B ∈ {0, 1}`.

**Canonical measurement condition:**
- Marginal sweep: `p ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}`, where
  `p = P(A = 1) = P(B = 1)` and `A`, `B` are independent.
- Canonical marginal: `p = 0.5` (balanced features; XOR is balanced and the
  best linear probe is at its weakest, ~0.75 accuracy).
- `1000` tokens per slice, fixed seed `42` for reproducibility.

`generate(seed)` is deterministic: the same seed yields byte-identical batches.

## Model function signature

```python
def model_fn(tokens: np.ndarray) -> np.ndarray:
    """
    Args:
        tokens: (N, 4) int array — one row per example, vocabulary as above.

    Returns:
        logits: (N,) float array — one logit per example. The prediction is
                XOR = 1 iff logit > 0.
    """
```

The attempt implements `model_fn` as any mapping it likes (a trained
transformer, a hand-built circuit, a lookup, anything). The benchmark only
evaluates the *predictions* derived from the returned logits; it does not
constrain the internal architecture. `evaluate` validates that each return value
is a finite real vector of the right length and raises `ValueError` otherwise.

## Payload contract

`task.evaluate(model_fn)` runs the model across the full marginal sweep and
returns a dict with exactly these keys:

```python
{
    "version": 1,                 # payload contract version
    "canonical_p": 0.5,           # the marginal at which the headline is reported
    "n_per_slice": 1000,          # tokens evaluated per sweep point
    "sweep": [                    # one record per marginal in the sweep
        {
            "p": float,                  # marginal P(A=1)=P(B=1) for this slice
            "accuracy": float,           # model accuracy on A XOR B, in [0, 1]
            "baseline_accuracy": float,  # best linear-probe accuracy, in [0, 1]
            "frac_xor1": float,          # empirical fraction with label XOR=1
            "n": int,                    # number of tokens in this slice
        }
        ...                              # exactly 9 records, p ascending
    ],
}
```

**`accuracy`** is `mean(prediction == label)` where `prediction = (logit > 0)`.

**`baseline_accuracy`** is the *best linear probe* on this slice's data, not the
constant/majority predictor. XOR and XNOR are the only two boolean functions of
`(A, B)` that are not linearly separable over the one-hot features; the baseline
enumerates the other 14 boolean predictors and takes the maximum empirical
accuracy. This floor is always `≥ max(P(XOR=1), P(XOR=0))` and in expectation
equals `1 - min(p, 1-p)**2` (`0.75` at `p = 0.5`, `~0.99` at `p = 0.1`). A
genuine XOR mechanism must beat this stronger floor, not merely the majority
class.

## Metrics

`benchmark.score(payload)` returns a flat dict. Let `key = _fmt_p(p)` render the
marginal in `0p5`-form (`0.1 → 0p1`, `0.5 → 0p5`).

| Metric | Formula | Direction | Interpretation |
|--------|---------|-----------|----------------|
| `version` | `payload["version"]` | — | Contract version for dashboard filtering |
| `xor_accuracy_p_<key>` | `sweep[p]["accuracy"]` | **Bigger is better** | Per-slice model accuracy at marginal `p` |
| `linear_baseline_accuracy_p_<key>` | `sweep[p]["baseline_accuracy"]` | — | Best linear probe at marginal `p` |
| `lift_over_linear_p_<key>` | `accuracy - baseline_accuracy` | **Bigger is better** | How far the model beats the linear floor at `p` |
| `xor_gap_capture_p_<key>` | `clamp01((accuracy - base) / (1 - base))`, or `1.0` if `base ≥ 1.0` | **Bigger is better** | Fraction of the above-baseline headroom captured at `p`, in `[0, 1]` |
| `xor_accuracy_canonical` | `xor_accuracy_p_0p5` | **Bigger is better** | Headline accuracy at the canonical marginal |
| `linear_baseline_accuracy_canonical` | `linear_baseline_accuracy_p_0p5` | — | Best linear probe at the canonical marginal |
| `lift_over_linear_canonical` | `lift_over_linear_p_0p5` | **Bigger is better** | Canonical lift over the linear floor |
| `worst_slice_accuracy` | `min` of `accuracy` over the sweep | **Bigger is better** | Robustness: weakest slice |
| `xor_robustness` | `mean` of `xor_gap_capture_p_<key>` over the sweep | **Bigger is better** | **Headline summary.** Mean fraction of above-baseline headroom captured across all marginals, in `[0, 1]`. `1.0` means perfect XOR everywhere; `0.0` means no better than the linear floor. |

**Edge cases.** When a slice's baseline is degenerate (`base ≥ 1.0`, only
possible if XOR is constant on the slice, which it is not for `p ∈ (0, 1)`),
`xor_gap_capture` is defined as `1.0` rather than dividing by a zero gap. The
sweep mean and `worst_slice` reductions are guarded against an empty sweep.

### Pipeline hook

`benchmark.is_obviously_broken(metrics) -> bool` returns `True` (failing the
attempt and skipping the jury) when the metrics are mechanically degenerate:
any `NaN`/`inf`, the canonical accuracy not clearing the canonical linear
baseline by at least `0.05`, or `xor_robustness ≤ 0`. It never returns `True`
for a borderline-but-real result.

## Bump procedure

`VERSION` in `benchmark.py` and `task.py` and the `version` in the payload
**must** be bumped together when:
- Any metric formula changes.
- Payload keys are added/removed/retyped.
- The canonical marginal (`CANONICAL_P`) or the sweep values (`SWEEP_PS`)
  change.
- The token vocabulary or encoding changes.

`task.SWEEP_PS` and `benchmark.SWEEP_PS` must stay identical (same values, same
ascending order); `benchmark.score` validates this and the payload `version`,
raising if they drift.

Adding a new per-slice metric derived from data already in the sweep does
**not** require a bump. After bumping, update this README's "Payload contract"
and "Metrics" sections in the same commit.
