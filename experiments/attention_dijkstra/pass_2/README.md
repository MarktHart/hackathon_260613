**What I did** (hand_built / interp)

This is a **hand-built soft-min attention circuit** — `base_model.py`'s
self-attention block with three edits and no learned parameters. (1) The `QKᵀ`
logits are replaced by the negated relaxation cost `−β·(dᵤ + w_{u→v})`, so the
softmax over source positions `u` becomes a *soft-argmin over predecessors*.
(2) The value read-out at the softmax is the **soft-min** `−1/β·logsumexp(−β·(dᵤ
+ w_{u→v}))` — a min-plus matmul, the genuine soft relaxation Dijkstra/Bellman–Ford
perform, **not** the previous attempt's hard `torch.min`. (3) The block is
weight-tied and applied recurrently for `n−1` hops (the Bellman–Ford bound, so
it adapts to graph diameter instead of a fixed 10), and the MLP is dropped.
`β` is the attention temperature: `β→∞` recovers exact hard Dijkstra, finite `β`
is true softmax attention. All compute is torch `float64` on CUDA. The model
reaches `distance_accuracy ≈ 1.0` across the 8→64 sweep while the no-propagation
one-hop baseline collapses (≈0.46 → 0.06).

**Why this visualisation**

Four panels each check a distinct claim the goal makes. (1) *Faithfulness* —
accuracy vs **number of relaxation hops**, one line per graph size: knocking out
hops (the only mechanism here) drops accuracy back to the one-hop baseline
dashed line, and larger graphs visibly need more hops, which is the causal
evidence that propagation depth — not memorisation — does the work, and why
`n−1` adaptive hops is the right operating range. (2) *Correctness/strawman* — a
predicted-vs-true scatter where soft-min points sit on `y=x` while the orange
one-hop baseline scatters above it. (3) *Temperature* — a β sweep showing low
(soft) β underestimates distances and high (sharp) β is exact, proving the
mechanism is a soft-min with a temperature knob rather than a hard min. (4)
*Mechanism* — the converged attention matrix `softmax(−β·cost)`, whose argmax is
the recovered shortest-path predecessor tree. The Benchmark tab tracks
`dijkstra_robustness` and canonical accuracy across attempts.
