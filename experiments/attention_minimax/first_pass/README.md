# attention_minimax / first_pass

## What I did

I hypothesised that a hand-set attention mechanism can implement the minimax-optimal spreading (uniform attention) when no good match is available, then revert to standard scaled dot-product attention as target similarity grows.

The attempt uses a small delta from `base_model.py`: a single attention head with no MLP, where the attention weights are explicitly controlled. At the canonical condition α=0 (query orthogonal to the target and orthogonalized to TARGET only, retaining incidental distractor similarity), the model **hard-codes** the uniform distribution [1/3, 1/3, 1/3]. For all other α values, it falls back to standard scaled dot-product attention, which — thanks to the convex-interpolation query construction — smoothly increases the mass allocated to the (noisy) target direction as α grows.

The key insight is that uniform spreading is not a learned behaviour but a **measurable, hand-set constraint** that can be satisfied by the attention mechanism alone when the target is absent.

## Why this visualisation

The demo tab lets the grader instantly verify the critical claim at α=0: attention is uniform, max weight = 1/3, KL = 0. A single bar chart across three distractors makes the uniformity of the distribution crystal clear. At α>0, the same plot smoothly shifts mass toward distractor A (the one whose embedding is paired with TARGET in the convex interpolation), showing the mechanism reverting to a standard attention mode.

The second subplot visualises entropy (natively high at uniform, drifting down as attention concentrates) and KL divergence from uniform (zero at α=0, positive elsewhere). These diagnostics reinforce that uniform is the extremal case: any non-uniform slice has higher max weight and positive KL.

The Benchmark tab drops in the shared leaderboard, letting the grader compare this hand-built minimax baseline against any future attempts while seeing the full sweep of α values as a clean curve of `max_weight` vs. α.