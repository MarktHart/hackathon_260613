## What I did

**Hand-built attention circuit** (no training, no Python branching). I express
logical OR as a **log-sum-exp soft maximum** — the operation that already lives
inside softmax attention — so the max is *computed*, not looked up. For any
query `q` and key matrix `K` (channels = the keys themselves) one fixed formula
runs:

```
s[j] = (1/beta) * logsumexp_c( log_softmax_c(gamma * (q . k_c)) + beta * (k_c . k_j) )
```

`gamma*(q·k_c)` is a query→channel gate (first-order attention: which
key-directions the query expresses); `beta*(k_c·k_j)` is each channel's key→key
footprint; `logsumexp/beta` is a soft maximum over the gated channels. This is a
small delta from `base_model.py` attention: the ordinary `softmax(q·K)` readout
is replaced by a gated, two-term soft-max over key-derived channels. For a single
feature query `q_A` only channel A gates on, recovering the ordinary score
`q_A·K`; for the balanced superposition `q_AB` both channels gate on equally, so
the readout becomes `max(q_A·k_j, q_B·k_j)` = OR. The **identical callable**
handles `q_A`, `q_B` and `q_AB` — unlike pass_3, there is no `if`-branch on
query identity and no `np.maximum` lookup; the max emerges from the soft-max as
`beta` grows. Because the per-slice directions are re-randomized, the circuit
reads its channel directions from `K` (the task guarantees the signal keys *are*
the query directions — the only available source). All compute runs in torch on
`cuda`. Result: `or_sharpness_canonical ≈ 0.98`, flat across the cosine sweep
(`superposition_robustness ≈ 1.0`).

**Faithfulness / causal evidence.** This is a synthetic circuit (no trained
model), so I verify the mechanism by **ablating its own components** at the
canonical anchor: (i) removing the soft-max (`beta→0`, a plain weighted average)
drops sharpness to ~0.5 — the max disappears; (ii) removing the gate (`gamma=0`)
lets the 62 noise channels in and noise leakage jumps toward 1.0 — signal/noise
separation disappears; (iii) the plain-linear superposition `q_AB·K` reaches only
0.707. Each ablation breaks a different desideratum, which is the causal claim:
the soft-max produces the OR, the gate produces the focus. On a real trained
model the analogous check would patch out the high-temperature soft-max head /
zero the query→channel gate and watch combined-query OR collapse to a linear
blend.

## Why this visualisation

Three panels, each checking one claim. **Panel A (sharpness vs cos)** puts the OR
circuit against a *real failing strawman* — plain-linear superposition, which
sits at `sqrt((1+cos)/2)` (0.707 at the orthogonal anchor, only reaching 1.0 when
the two queries collapse into one). The OR circuit stays flat near the ideal 1.0
line where the strawman fails; the gap at low cos is the whole result. This fixes
pass_3's mistake of using the benchmark's `s_A+s_B` reference (an *upper oracle*
that rises to 2.0) as if it were a strawman. **Panel B (ablation bars)** shows
sharpness *and* noise leakage for full / no-soft-max / no-gate / plain-linear, so
a human can see each component is load-bearing and which metric it controls.
**Panel C (beta sweep)** plots sharpness at the anchor as the soft-max
temperature rises, tracing the smooth `average → hard max` transition with the
0.707 strawman and 1.0 ideal as reference lines — visual proof that the OR is the
soft-max, not a lookup.
