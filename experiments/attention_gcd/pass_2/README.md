# attention_gcd / pass_2

## What I did
This is a **hand_built (interp)** attempt — no training, every weight set by
hand, expressed as torch tensors on CUDA. It is `base_model.py` collapsed to a
single attention layer whose `SEP` output writes a hand-designed feature map
into the residual: (1) a **divisibility feature** `c[d]=[d|a and d|b]` for
`d=1..MAX_N` (the common divisors of `a,b`), then (2) a suffix-OR (reverse
cumulative max) giving a **thermometer code** `t[k]=[gcd(a,b)≥k]`. Because
`gcd=Σ_k t[k]`, a linear counting probe recovers gcd at R²≈1 / acc=1, while the
same probe on raw `[a,b]` is near-useless (baseline R²≈0). Over first_pass
(which scored *good* but was weak on faithfulness and operating range, and
whose trained variant timed out) pass_2 adds two things the rubric asks for: a
**causal ablation** that zeros the thermometer subspace and re-runs the
evaluator — decodability collapses to the raw-input baseline, proving the
residual mechanism is *causally* responsible — and a **MAX_N scale sweep**
(10→1000, with `d_model=MAX_N+pad` so it never breaks silently above 128).

## Why this visualisation
The headline question is "is gcd *linearly* decodable from the residual, beyond
raw inputs?", so the Demo leads with a **predicted-vs-true scatter** on `y=x`:
points on the diagonal *are* the R²≈1 claim. The second panel pairs the
**ablation bar chart** (full circuit vs thermometer-knocked-out vs raw-`[a,b]`
baseline) with the **R²-vs-MAX_N scale curve**. The ablation is the falsifiable
faithfulness test — if the red "ablated" bars stayed high, the mechanism
wouldn't be what carries gcd; instead they drop to the grey baseline. The
log-x scale curve shows the circuit holding across two orders of magnitude
while the baseline stays flat at zero. The Benchmark tab drops in the shared
leaderboard so iteration shows as a moving line.
