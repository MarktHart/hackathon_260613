# attention_gcd / first_pass

## What I did
This is a **hand_built** (interp) attempt — no training, every weight set by
hand. It is `base_model.py` collapsed to a single attention layer whose
SEP-position output writes a hand-designed feature map into the residual
stream. The mechanism is a two-step common-divisor circuit: (1) a
**divisibility feature** `c[d] = [d | a and d | b]` for every scale
`d = 1..MAX_N` (the common divisors of `a, b`), then (2) a suffix-OR
(reverse cumulative max) turning it into a **thermometer code**
`t[k] = [gcd(a,b) ≥ k]`. Because `gcd = Σ_k t[k]`, a single linear counting
probe recovers gcd at R² ≈ 1 / accuracy = 1, while the same probe on the raw
operands `[a, b]` is near-useless (gcd is violently non-linear in `a, b`,
baseline R² ≈ 0). Head-0's SEP→operand attention weight is additionally
scaled by the normalised gcd, so the attention pattern itself correlates with
gcd (corr ≈ 1.0 vs the `a+b` baseline ≈ 0.09). All compute runs in torch on
CUDA.

## Why this visualisation
The headline question is "is gcd *linearly* decodable from the residual,
beyond raw inputs?" — so the Demo leads with a **predicted-vs-true gcd
scatter** against the `y=x` line: points on the diagonal *are* the R²≈1 claim,
visually. Next to it, a **thermometer heatmap** (rows sorted by gcd) shows the
staircase `t[k] = [gcd ≥ k]` that makes gcd a linear count — you can see the
mechanism, not just its score. A grouped **circuit-vs-baseline bar chart**
(resid R², decode acc, attn corr) puts the falsifiable comparison on one axis:
if the grey raw-`[a,b]` bars matched the green circuit bars, there would be no
mechanism to interpret. The Benchmark tab drops in the shared leaderboard so
iteration shows as a moving line.
