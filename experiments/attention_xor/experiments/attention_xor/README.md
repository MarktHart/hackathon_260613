# attention_xor

## Question

Can a mechanism compute the **XOR** of two binary features `A` and `B` —
predicting `1` iff *exactly one* of them is present — and hold that accuracy up
across the full range of feature marginals, *without collapsing to a linear
approximation*? XOR is the canonical non-linearly-separable function: no linear
threshold over the one-hot `(A, B)` features can realise it. A genuine XOR
mechanism must beat the best linear probe at every marginal.

## Setup

**Synthetic generator.** No trained model is required by the task. `task.py`
constructs token batches encoding two independent Bernoulli features and the
label `A XOR B`.

**Token vocabulary** (see `task.CLS`/`task.SEP`):

| id | meaning |
|----|---------|
| 0  | `CLS`           |
| 1  | `A0` (feature A = 0) |
| 2  | `A1` (feature A = 1) |
| 3  | `B0` (feature B = 0) |
| 4  | `B1` (feature B = 1) |
| 5  | `SEP`           |

Each example is the length-4 token row `[CLS, A_tok, B_tok, SEP]`, where
`A_tok ∈ {1, 2}` encodes `A ∈ {0, 1}` and `B_tok ∈ {3, 4}` encodes
`B ∈ {0, 1}`. The label is `A XOR B`.

**Marginal sweep.** For each `p` in the sweep, `A` and `B` are independent
`Bernoulli(p)` draws (`p = P(A=1) = P(B=1)`). As `p` moves away from `0.5`, one
label class dominates, so a constant or linear predictor scores higher — the
sweep stresses the mechanism against an increasingly strong linear floor.

**Canonical measurement condition:**
- Sweep: `p ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}`
- Canonical marginal: `p = 0.5` (balanced; the linear floor is at its lowest, ~0.75)
- `1000` tokens per slice (`N_PER_SLICE`)
- Fixed seed `42` (`EVAL_SEED`) for reproducibility

`task.generate(seed)` is **deterministic**: the same seed yields byte-identical
batches. It returns one `Batch` (tokens, labels, `p`) per sweep value.

## Model function signature

```python
def model_fn(tokens: np.ndarray) -> np.ndarray:
    """
    Args:
        tokens: (N, 4) int array — one row [CLS, A_tok, B_tok, SEP] per example.
                A_tok in {1, 2} -> A in {0, 1};  B_tok in {3, 4} -> B in {0, 1}.

    Returns:
        logits: (N,) float array. The prediction is XOR=1 iff logit > 0.0.
    """
```

The attempt implements `model_fn` as any mapping (differentiable or not). The
benchmark evaluates only the returned logits; it does not constrain the internal
architecture. `evaluate` raises `ValueError` if `model_fn` returns the wrong
shape or any non-finite value.

## Payload contract

`task.evaluate(model_fn, *, seed=42)` returns a dict with exactly these keys:

```python
{
    "version": 1,           # payload contract version (== benchmark.VERSION)
    "canonical_p": 0.5,     # the marginal at which the headline canonical is reported
    "n_per_slice": 1000,    # tokens per sweep slice
    "sweep": [              # one record per p in the sweep, ascending
        {
            "p": float,                 # marginal P(A=1) = P(B=1) for this slice
            "accuracy": float,          # mean(model prediction == label), in [0, 1]
            "baseline_accuracy": float, # best-linear-probe accuracy on this slice (see Metrics)
            "frac_xor1": float,         # observed fraction of label==1 in this slice
            "n": int,                   # number of tokens in this slice
        }
        ...
    ],
}
```

`benchmark.score` validates that `sweep` has exactly `len(SWEEP_PS) == 9`
records and that each record's `p` matches the expected sweep value (to `1e-9`),
that `version` matches `benchmark.VERSION`, and that every required record key is
present and numeric/finite.

**Baseline definition.** `baseline_accuracy` is the accuracy of the *best linear
probe* over the one-hot `(A, B)` features on that slice's data — i.e. the maximum
in-sample accuracy across all 16 boolean functions of `(A, B)` **except** the two
that are not linearly separable, XOR and XNOR. This is a strictly stronger floor
than the majority-class constant (it can predict `A OR B`, a single corner,
etc.), with expected value `1 − min(p, 1−p)²` (≈0.75 at `p=0.5`, ≈0.99 at the
extremes). A real XOR mechanism must beat *this*, not merely the constant
predictor.

## Metrics

`benchmark.score(payload)` returns a flat dict. Let `key = _fmt_p(p)` render the
marginal in `0p5`-form (`0.5 → 0p5`).

| Metric | Formula | Direction | Interpretation |
|--------|---------|-----------|----------------|
| `version` | `payload["version"]` | — | Contract version for dashboard filtering |
| `xor_accuracy_p_{key}` | `sweep[p]["accuracy"]` | **Bigger is better** | Per-slice model accuracy at marginal `p` |
| `linear_baseline_accuracy_p_{key}` | `sweep[p]["baseline_accuracy"]` | — | Per-slice best-linear-probe floor at `p` |
| `lift_over_linear_p_{key}` | `accuracy − baseline_accuracy` | **Bigger is better** | Per-slice margin over the linear floor (can be negative) |
| `xor_gap_capture_p_{key}` | `clamp01((accuracy − base) / (1 − base))`; `1.0` if `base ≥ 1.0` | **Bigger is better** | Fraction of the above-floor headroom captured at `p`, in `[0, 1]` |
| `xor_accuracy_canonical` | `xor_accuracy_p_0p5` | **Bigger is better** | Headline accuracy at the canonical marginal |
| `linear_baseline_accuracy_canonical` | `linear_baseline_accuracy_p_0p5` | — | Linear floor at the canonical marginal |
| `lift_over_linear_canonical` | `lift_over_linear_p_0p5` | **Bigger is better** | Canonical margin over the linear floor |
| `worst_slice_accuracy` | `min over slices of accuracy` | **Bigger is better** | Robustness floor across the sweep |
| `xor_robustness` | `mean over slices of xor_gap_capture_p_{key}` | **Bigger is better** | **Headline summary**: mean fraction of above-linear headroom captured across the whole sweep, in `[0, 1]`. `0` means no better than the linear floor anywhere; `1` means perfect XOR everywhere. |

**Headline summary:** `xor_robustness`. **Per-slice values:** the four
`*_p_{key}` families. **Reference baseline:** `linear_baseline_accuracy_*` under
identical conditions.

**Edge cases handled in `score()`:**
- Zero headroom (`base ≥ 1.0`, only possible degenerately) → `xor_gap_capture`
  is defined as `1.0` rather than dividing by zero.
- `xor_gap_capture` is clamped to `[0, 1]`, so a below-floor model scores `0`,
  not a negative number.
- Empty sweep / empty accuracy list → aggregates default to `0.0` (the length
  check makes this unreachable in practice, but it is guarded).
- Non-numeric or non-finite `accuracy`/`baseline_accuracy` → `ValueError`.

### Pipeline hook

`benchmark.is_obviously_broken(metrics)` returns `True` (skip the jury) when any
metric is NaN/inf, when `xor_accuracy_canonical ≤ linear_baseline_accuracy_canonical + 0.05`
(no clear margin over the ~0.75 linear floor at `p=0.5`), or when
`xor_robustness ≤ 0`. It never returns `True` for a borderline-but-real result.

## Bump procedure

- `VERSION` in `benchmark.py` and `VERSION` in `task.py` (and `version` in the
  payload) **must** be bumped together when:
  - Any metric formula changes;
  - Payload keys are added/removed/retyped;
  - The canonical marginal `CANONICAL_P`, the sweep `SWEEP_PS`, `N_PER_SLICE`,
    `EVAL_SEED`, or the token vocabulary changes;
  - The linear-baseline definition changes.
- `task.VERSION` and `benchmark.VERSION` are kept in lock-step; `score()`
  rejects any payload whose `version` does not equal `benchmark.VERSION`, so a
  missed bump fails loudly.
- Adding a brand-new metric that does not change existing ones does **not**
  require a bump.
- After bumping, update this README's "Payload contract" and "Metrics" sections
  in the same commit.
