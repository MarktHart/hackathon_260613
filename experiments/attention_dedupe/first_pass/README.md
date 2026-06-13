# Attention Deduplication — hand-built duplicate-token head

## What I did

This is a **hand_built** attempt: no training, weights set by hand. It is
`base_model.py` cut down to a *single causal attention layer with no MLP*, where
the learned bilinear QK score is replaced by an exact rule. For query position
`q` holding token `t`, the score on key `k` is `BIG + REC·k` if `k < q` and
`tok_k == t` (an earlier occurrence of the same token), `SELF` on the diagonal
`k == q`, and `−inf` otherwise. A softmax over keys then concentrates a
duplicate query's mass on the **most recent earlier position with the same
token** — the `REC·k` recency ramp makes the largest matching index win, and
`BIG ≫ SELF` makes any real match beat the diagonal. First-seen tokens have no
earlier match, so only the diagonal survives and they stay on themselves, which
is the desired behaviour for novel tokens. Everything runs on CUDA via torch;
the circuit is exact, so it should hit ≈1.0 dedup mass/accuracy across the whole
dup-rate sweep, far above the uniform-causal baseline (`1/(i+1)`).

## Why this visualisation

The headline claim is "mass lands on the previous occurrence", so the Demo tab
shows the raw `(L×L)` attention heatmap for a chosen sequence with the
ground-truth previous-occurrence index marked by a ▲ on each duplicate query
row. If the bright cell sits on the ▲ for every repeated token and on the
diagonal for first-seen tokens, the mechanism is correct *by eye* — no metric
needed. The second chart puts `dedup_mass` next to the uniform-causal baseline
across `dup_rate ∈ {0.1,0.3,0.5,0.7}`, which is exactly the benchmark's
bigger-is-better comparison: it makes the *lift over the strawman* legible at
every density rather than collapsing it to one number. The Benchmark tab carries
the shared leaderboard so this hand-built head can be compared against future
trained or interp attempts.
