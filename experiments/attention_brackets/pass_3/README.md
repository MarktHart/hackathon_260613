## What I did

**Type: hand_built** (a hand-set attention circuit, no training). I express
bracket matching as a single QK attention head that is a small delta from
`base_model.py` self-attention. The trick is a structural **nesting-level**
feature: `level[i]` is the stack height token `i` sits at, a running signed
cumsum of the token stream (`+1` for `(`, `−1` for `)`) — exactly what one
preceding causal head over a signed token embedding would compute, and never
the ground-truth match array. Queries (closers) and keys (openers) are one-hot
encodings of `level`, so `Q·K = 1` iff a closer and opener share a nesting
level — the stack-matching condition. A tiny recency bias `ALPHA · position`
breaks ties toward the *most recent* same-level opener, which is precisely the
one the stack pops, and the level-match weight `C ≫ ALPHA·L` guarantees a
same-level opener always outscores any other key. The result is
`score[i,j] = C·[level_i == level_j ∧ j opens ∧ i closes] + ALPHA·j`, causal-
masked and softmaxed. This fixes pass_2's broken scalar `q·k`, whose dot
product peaked on the *last* position rather than the matching opener.

**Faithfulness note.** This is a synthetic hand-built circuit, so there is no
trained model to ablate. The honest causal check is built into the construction
and shown in the demo: knock out the level feature and only the recency term
survives, which *is* the nearest-opener heuristic — and the demo shows that
heuristic collapsing on outer brackets as depth grows. So the level-equality
term is provably the component doing the stack matching. A model-level version
of this claim would patch the level subspace out of a trained head's keys and
watch deep-nesting accuracy fall to the nearest-opener line.

## Why this visualisation

The **heatmap** puts query (closing) positions on the y-axis and key (opener
candidate) positions on the x-axis, with the parser's true matching opener
ringed in cyan. The claim "attention lands on the matched opener, not a
positional shortcut" is true exactly when every bright cell sits inside its cyan
ring — directly readable, including the long off-diagonal jumps an outer bracket
must make across balanced inner content.

The **depth-sweep bars** are the baseline comparison the goal asks for: argmax
accuracy and normalised lift over uniform, for the stack head vs. nearest-opener
vs. random, grouped by nesting depth (d1–d5). The stack head holds ≈1.0 across
all depths while nearest-opener — correct at d1 — visibly decays as deeper
nesting forces outer closers past intervening openers, and random sits at the
floor. That divergence is the testable statement: the mechanism works *where the
cheap heuristic fails*.
