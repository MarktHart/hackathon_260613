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
