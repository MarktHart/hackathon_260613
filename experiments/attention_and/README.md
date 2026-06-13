# Attention as a soft AND gate

## Goal
Show how softmax attention can approximate an AND gate.

## Exmample
### Setup

Scaled dot-product attention assigns to each key/value index $i$ a weight

$$
w_i \;=\; \frac{\exp(q \cdot k_i / \sqrt{d})}{\sum_j \exp(q \cdot k_j / \sqrt{d})}.
$$

The output is $\sum_i w_i \, v_i$.

The key (no pun intended) move is to feed in a query that is a **superposition** of two concept directions $q_A$ and $q_B$:

$$
q \;=\; q_A + q_B.
$$

You can think of $q_A$ as "asks: does this token have feature $A$?" and $q_B$ as "asks: does this token have feature $B$?".

### Softmax turns sums into products

The dot product is linear in $q$, so superposition in $q$ becomes a *sum* in the score:

$$
q \cdot k_i \;=\; q_A \cdot k_i + q_B \cdot k_i.
$$

But the softmax exponentiates that score. Exponential of a sum = product of exponentials:

$$
\exp(q \cdot k_i) \;=\; \exp(q_A \cdot k_i)\,\exp(q_B \cdot k_i).
$$

Define
$\alpha_i := \exp(q_A \cdot k_i / \sqrt{d})$ ("soft indicator that $k_i$ has feature $A$") and
$\beta_i := \exp(q_B \cdot k_i / \sqrt{d})$ ("soft indicator that $k_i$ has feature $B$").

Then the (unnormalized) attention mass on token $i$ is

$$
\tilde w_i \;=\; \alpha_i \cdot \beta_i,
$$

i.e. the **product** of how-much-$A$ and how-much-$B$ that token has. The normalizer $Z = \sum_j \alpha_j \beta_j$ just turns this into a probability.

A product is the natural soft AND: it is large only when *both* factors are large; if either $\alpha_i$ or $\beta_i$ is small, $\tilde w_i$ collapses. A pure linear layer (no exp) would only give you $\alpha_i + \beta_i$-like behavior, which is OR-ish — one large factor is enough.

### Concrete example

Three keys, scores in $(q_A \cdot k_i,\; q_B \cdot k_i)$ form, scale $1/\sqrt d$ absorbed:

| token | $A$-match | $B$-match | sum (linear) | product $\alpha_i\beta_i$ |
|-------|-----------|-----------|--------------|---------------------------|
| 1: only A  | 6 | 0 | 6 | $e^{6}\approx 403$ |
| 2: only B  | 0 | 6 | 6 | $e^{6}\approx 403$ |
| 3: both    | 4 | 4 | 8 | $e^{8}\approx 2981$ |

Without the exponential, token 3 barely edges out 1 and 2 (8 vs 6). After softmax, token 3 grabs $\approx 79\%$ of the mass while tokens 1 and 2 share the rest — the head sharply prefers the conjunction. Push the contrast a bit (e.g. token 3 at $(5,5)$ vs token 1 at $(7,0)$) and the AND-vs-OR gap grows: $e^{10}$ beats $e^{7}$ by $\sim 20\times$, even though their linear scores tie.

### Why this is "kinda an AND"

- **Multiplicative gating.** $\tilde w_i = \alpha_i \beta_i$ is exactly the form of a soft AND on the two soft indicators.
- **Vanishes if either input vanishes.** If $k_i$ has no $B$-component, then $\beta_i \to 1$ (baseline, not small) — caveat: it's only AND-like when $q_B \cdot k_i$ can become *negative* for non-$B$ tokens. So the gate is sharp to the extent the unembed directions $q_A, q_B$ separate matching keys from non-matching ones with positive vs. negative scores.
- **Generalizes to $n$-ary AND.** Stacking $q = q_{A_1} + \dots + q_{A_n}$ gives $\tilde w_i = \prod_k \alpha_i^{(k)}$, i.e. attention can soft-AND arbitrarily many concepts in a single head, up to the precision allowed by superposition and the QK rank.
- **Normalizer = competitive read-out.** The softmax denominator turns the multiplicative score into a winner-take-most distribution, so the value vector of the AND-matching token gets routed through.

### One-line summary

Because the softmax exponentiates a linear score, a query that is a sum of concept directions becomes, after $\exp$, a **product of per-concept match scores** — and a product of soft indicators is a soft AND.

## Benchmark (v2)

Every attempt is scored by `benchmark.py` in this directory. The canonical
measurement scale is **`scale = 1.0`** — the same value every attempt must
use so the numbers are comparable. The benchmark sweeps `cos(q_A, q_B)` so
attempts that only work under orthogonal concept directions show up as fragile
under superposition.

### Payload contract

Hand `agentic.experiments.record_benchmark(__file__, run_dir, payload)` a dict
with these keys:

| key | type | meaning |
|-----|------|---------|
| `sweep`                  | `list[dict]`       | one record per cosine slice — see below |
| `both_label`             | `str`              | which key in the per-slice weight dicts is the AND-target token |
| `single_feature_labels`  | `list[str]`        | keys of the A-only and B-only tokens |
| `canonical_scale`        | `float`            | record-keeping; should be `1.0` |

Each `sweep` record:

| key | type | meaning |
|-----|------|---------|
| `cosine`           | `float`             | `cos(q_A, q_B)` for this slice; pick at least `0.0` and one positive value |
| `softmax_weights`  | `dict[str, float]`  | per-token softmax mass at this cosine, canonical scale (must sum to ~1) |
| `linear_weights`   | `dict[str, float]`  | per-token linear-baseline mass at the same setting |

The reference sweep uses `cosine ∈ {0.0, 0.3, 0.5, 0.7, 0.9}`. Attempts can
add finer-grained slices but must include `0.0` (the orthogonal anchor for
`*_canonical` metrics) and at least one positive cosine (so
`superposition_robustness` is computable).

### Metrics

Per slice (one named scalar per cosine value `c`, e.g. `cos_0p7`):

| metric | meaning |
|--------|---------|
| `and_sharpness_cos_<c>`              | `softmax[both] / mean(softmax[single])` at that cosine — the core curve |
| `linear_baseline_sharpness_cos_<c>`  | same ratio for the linear baseline — the no-`exp` ceiling |

Summary (the headline numbers):

| metric | meaning |
|--------|---------|
| `superposition_robustness`     | `and_sharpness` at highest cosine ÷ at lowest. `1.0` = method survives superposition; `→ 0` = collapses. |
| `and_sharpness_canonical`      | `and_sharpness` at the lowest cosine (most orthogonal) — the original "does it AND-gate at all?" |
| `softmax_both_mass_canonical`  | softmax mass on the `both` token at the lowest cosine |

Bump `VERSION` in `benchmark.py` if you change the formulae — the dashboard
filters to the latest version so older `benchmark.json` files stay readable
without polluting the active series.
