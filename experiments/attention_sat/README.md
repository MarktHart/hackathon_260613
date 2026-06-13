# attention_sat: Attention Saturation Robustness

## Question

How well does an interpretability method detect and characterize **attention saturation** — the phenomenon where attention logits grow large in magnitude, causing softmax to concentrate probability mass on a few positions (effectively "hard" attention), and where gradients vanish or explode?

This goal asks: *Does the method correctly identify the saturation regime, quantify the degree of saturation, and distinguish true mechanistic saturation from mere high-confidence attention?*

---

## Setup

**Synthetic generator only** — no trained model required. We construct a controlled attention module with a single head where we can analytically control the logit scale (inverse temperature) and measure the resulting saturation.

- **Generator**: `task.generate(seed)` produces a `Batch` containing:
  - `q`: query vectors, shape `(batch, seq_len, d_head)`
  - `k`: key vectors, shape `(batch, seq_len, d_head)`
  - `v`: value vectors, shape `(batch, seq_len, d_head)`
  - `logit_scales`: list of scalar multipliers applied to `q @ k.T` before softmax. Each scale defines one measurement condition.
  - `causal_mask`: optional boolean causal mask.

- **Canonical measurement condition**: `logit_scales = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]`. This spans the linear regime (scale ≈ 0.1), the transition (1–3), and deep saturation (≥ 30). The **canonical scale** for the headline metric is `logit_scale = 10.0`.

- **Model function signature** (the contract between attempt and task):
  ```python
  def model_fn(q: np.ndarray, k: np.ndarray, v: np.ndarray,
               logit_scale: float, causal_mask: np.ndarray | None) -> dict:
      """
      Returns a dict with keys:
          - 'attn_weights': np.ndarray, shape (batch, seq_len, seq_len)
          - 'attn_entropy': np.ndarray, shape (batch, seq_len)  # per-query entropy
          - 'saturation_score': float  # scalar summary for this scale
      """
  ```
  The attempt's `model_fn` receives one scale at a time; `task.evaluate` loops over `logit_scales`.

---

## Payload Contract

`task.evaluate(model_fn)` returns a dict with these keys:

| key | type | description |
|-----|------|-------------|
| `version` | int | Payload schema version (mirrors `benchmark.VERSION`). |
| `sweep` | list[dict] | One record per `logit_scale`. Each record contains: |
| | | `logit_scale`: float — the scale used |
| | | `attn_weights`: np.ndarray — shape (batch, seq_len, seq_len), attempt's weights |
| | | `attn_entropy`: np.ndarray — shape (batch, seq_len), attempt's per-query entropy |
| | | `saturation_score`: float — attempt's scalar summary |
| | | `max_attn_weight`: float — mean over (batch, query) of the attempt's per-query max weight |
| | | `mean_entropy`: float — mean of the attempt's `attn_entropy` over batch, seq |
| | | `ref_max_attn_weight`: float — same reduction on the **analytic reference** weights |
| | | `ref_mean_entropy`: float — mean of the **analytic reference** per-query entropy |
| `config` | dict | Fixed generation config: `batch`, `seq_len`, `d_head`, `seed` |

`benchmark.score` **requires** every per-record key above except `attn_weights`
and `attn_entropy` (those two are optional and currently carried for debugging /
future per-query metrics). All scalar fields are Python floats; the array
fields are `float32` NumPy arrays. The payload contains **no torch tensors**.

---

## Metrics

`benchmark.score(payload)` returns a flat dict of scalars. Metrics are **bigger-is-better** unless noted.

### Headline summary
- **`saturation_detection_auroc`** (float, [0, 1]): Area under the ROC curve for detecting "saturated" vs "non-saturated" regimes using the attempt's `saturation_score`. Ground truth: saturated ⇔ `logit_scale ≥ 10.0`. Higher = better detection.

Scale values are formatted as `<int>p<frac>` (e.g. `0.1 → 0p1`, `10.0 → 10p0`,
`100.0 → 100p0`).

### Per-slice values (one per `logit_scale` in the sweep)
- **`mean_entropy_logit_<scale>`** (float): The **attempt's** mean attention entropy at this scale.
- **`ref_mean_entropy_logit_<scale>`** (float): The **analytic reference** mean entropy at this scale (ground truth).
- **`max_weight_logit_<scale>`** (float): The **attempt's** mean per-query max attention weight at this scale.
- **`ref_max_weight_logit_<scale>`** (float): The **analytic reference** mean per-query max weight at this scale (ground truth).
- **`saturation_score_logit_<scale>`** (float): The attempt's reported `saturation_score` at this scale.

### Sweep-level fidelity
- **`entropy_correlation_sweep`** (float, [-1, 1]): Pearson correlation between the attempt's `mean_entropy` and the reference `ref_mean_entropy` across the seven scales. `1.0` means the attempt tracks the true entropy curve; `0.0` (e.g. a constant-entropy attempt) means it does not. Returns `0.0` for a constant series.

### Reference baselines
- **`linear_baseline_auroc`** (float): AUROC using `saturation_score = logit_scale` (no mechanistic insight). **Because the saturation label is `logit_scale ≥ 10`, scale is a perfect separator and this is always ≈ 1.0** — it is an *oracle* reference, not a strawman. A method cannot beat it; matching it (your AUROC ≈ 1.0) is the goal.
- **`entropy_baseline_auroc`** (float): AUROC using `-ref_mean_entropy` (true entropy drops under saturation) as the detector. Also an oracle upper bound (≈ 1.0).

### Derived
- **`lift_over_linear`** = `saturation_detection_auroc - linear_baseline_auroc` ∈ `[-1, 0]`. `0` = matched the oracle; negative = worse. Bigger is better.
- **`lift_over_entropy`** = `entropy_baseline_auroc - saturation_detection_auroc` ∈ `[0, 1]`. `0` = matched the oracle; smaller gap = closer to oracle.

---

## Bump Procedure

`benchmark.VERSION` is bumped when:
- Any metric formula changes.
- Payload keys are added/removed/retyped.
- Canonical `logit_scales` list changes.

Current `VERSION = 1`.