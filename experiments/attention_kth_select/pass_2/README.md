# attention_kth_select / pass_2

## What I did

**Attempt type: hand_built (no training).** This is `base_model.py` reduced to a
single self-attention head with hand-set Q/K weights and no MLP/value — the
metric scores the attention pattern only. The query is one fixed vector,
`q = β·onehot(99)`; keys are one-hot token embeddings; `softmax(K·q)` attends to
positions carrying the marker. Crucially, **all compute is per sequence** — the
fix for `first_pass`, which recovered `k` by averaging the marker mask across the
whole batch (`marker_mask.mean(dim=0)`), a cross-sequence statistic no real
attention head can compute and the reason its `attn_at_k=1.0` was an artefact.
Within a single sequence `k` is revealed only by the marker, which also collides
onto other positions with probability `r=1/V`, so the positions holding the
marker are *exchangeable* and the Bayes-optimal mass on `k` is
`E[1/(1+S)], S~Binomial(L-1, r) ≈ 0.86` — **not 1.0**. The submitted head hits
this analytic ceiling, and I add (a) a **causal ablation** — zeroing the query or
the marker key-channel collapses it to the uniform `1/L` baseline; (b) an
**operating-range sweep** over `r` showing empirical accuracy tracks the closed
form across two orders of magnitude; and (c) a **fixed-position strawman** that
nails `k=8` but fails at every other `k`, plus the old batch oracle kept only as
an explicitly-labelled *unfaithful* upper bound.

## Why this visualisation

The default **Sweep accuracy** view puts `attn_at_k` (mass on the correct
position — the exact thing the benchmark scores) on the y-axis against every `k`,
overlaying four methods, the analytic Bayes ceiling, and the uniform baseline.
It makes the whole claim legible at a glance: the faithful pointer is a flat line
*on the ceiling* (tracks every `k`), the fixed-position strawman spikes only at
`k=8` (can't track a varying target), and the oracle's flat 1.0 is visibly
*above* the ceiling — the tell-tale of cross-sequence cheating. The **Operating
range** view is the smallest artefact that, if it deviated, would break the claim
the ~0.86 is principled rather than a failure: empirical points sitting exactly on
the analytic `E[1/(1+S)]` curve prove the ceiling is understood and predicted, and
show precisely where collisions degrade it. The **Ablation** bars give the causal
check the rubric asks for — knock out the query or marker channel and selection
collapses to `1/L`. The Benchmark tab tracks `kth_select_accuracy_canonical` and
`lift_over_linear_baseline_canonical` across attempts.
