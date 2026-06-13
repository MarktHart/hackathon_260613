## What I did
This is a **hand_built** attempt (no training). The model is a single attention
head — the smallest delta from `base_model.py`: I keep one attention layer and
drop the MLP entirely, because the function is expressible as one set of
query/key scores. The head's logits over the `N×N` grid are the **negative A\*
f-value**, `f = g + h`, where `h` is the Manhattan distance to the goal and `g`
is the true shortest free-path distance from the agent, computed on the GPU by
iterative min-plus (BFS) relaxation in torch tensors on `cuda`. Attention is
`softmax(-β·f − ε·g)` over reachable free cells with the agent's own cell
suppressed; the `−ε·g` tie-breaker resolves the A\*-tied optimal-path cells in
favour of the immediate next step, so the argmax is the cell A\* expands next.
The previous attempt failed because it used the wrong `model_fn` signature
(`grid, agent_pos, goal_pos`) — this one matches `task.evaluate`'s
`model_fn(grids)→[B,N,N]` contract exactly.

## Why this visualisation
The Demo tab puts the **attention heatmap next to the A\* f-value heatmap** for
the same grid, so a human can directly check the claim that attention mass
tracks low-f cells. The agent (cyan), goal (green), optimal next neighbours
(white dashed), the attention argmax (red ★) and obstacles (×) are overlaid, so
"the peak falls on the cell A\* would expand next" is visually verifiable in one
glance. The second panel is the **baseline-comparison bar chart**: it shows
`heuristic_alignment` and `top1_optimal_rate` for the full `g+h` circuit versus
the `h`-only, `g`-only, and uniform ablations at the canonical density — the
testable claim is not "it aligns" but "it aligns where the strawmen don't". A
purely synthetic circuit has no model to ablate, so the ablation here knocks out
each *component of the mechanism itself* (remove `g`, remove `h`) and watches the
metric collapse, which is the faithfulness evidence appropriate to a hand-built
attempt.
