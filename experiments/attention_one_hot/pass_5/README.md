# attention_one_hot — pass_5

## What I did

This is a **hand-built** attempt (zero learned parameters): a single scaled
dot-product attention head, `softmax((keys @ query) / τ)` with `τ = 0.1`, which
is the smallest possible delta from `base_model.py` (one attention head, no MLP,
no positional encoding, no learned projections). On the task's idealised
orthogonal-noise data it produces near-perfect one-hot lookup (canonical
`target_attention ≈ 0.997` at `L=64`, robust across `L = 16…256`).

The prior attempt (pass_4) scored well on the mechanism but was held back on
the two dominant rubric items — **baseline comparison** and **faithfulness** —
because it only drew an *analytic* `1/L` line and ran no ablation. pass_5 fixes
exactly that. (1) It runs a **measured strawman suite** through the *same*
`task.evaluate` evaluator: a no-temperature softmax (`τ=1`), a no-attention
uniform head, and a **causally query-patched** head; all three collapse to
`≈1/L` while the method stays `≈1.0`. (2) The query patch is an
**activation-patching causal check** — corrupting the query activation (the wire
that carries the lookup key) destroys one-hot, proving the `q·k` dot product is
load-bearing rather than incidental. (3) A **query-key alignment sweep** under
*realistic, non-orthogonal* noise keys maps each variant's operating range and
breaking point, showing why the `exp` non-linearity and the sharp temperature
each matter. All compute runs in torch on CUDA.

## Why this visualisation

Two panels, each tied to one rubric claim.

**Panel A — measured baseline comparison (bar chart, `target_attention` at the
canonical `L=64`).** Bars for `method`, `ablate-τ`, `patch-query`, and
`no-attention` sit against the dashed `1/L` reference. The single comparison the
goal cares about — "does attention put its mass on the one true needle?" — is
read off directly: the method bar is ≈1.0, every strawman bar is on the floor.
This is the testable "X works while no-exp/no-temperature/no-attention doesn't"
statement, with real numbers from the shared evaluator rather than an analytic
curve. The `patch-query` bar doubles as the causal/faithfulness evidence: knock
out the query and the behaviour breaks.

**Panel B — operating range (line chart, `target_attention` vs query-key
alignment α).** Under realistic noise the variants separate: the method holds
near 1.0 until α drops low, the no-temperature and no-exp ablations degrade much
earlier, and at α→0 even the method falls to `1/L` — no key match, no lookup,
which is the correct causal signature. The x-axis is the one quantity that
controls whether the lookup *can* succeed; putting it against `target_attention`
shows both the robustness margin and where the mechanism is *supposed* to fail.
The Benchmark tab adds the cross-attempt leaderboard and metric history.
