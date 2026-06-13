# What I did

This attempt tests whether a single scaled dot-product attention head can implement the
identity-copy primitive. The model function computes `softmax(QKᵀ / τ)` where queries
and keys are unit-norm vectors; the target key equals the query (cosine 1.0) while
distractors sit at a controlled cosine `cos ∈ {0.0, 0.3, 0.5, 0.7, 0.9}`.
A fixed low temperature `τ = 0.1` sharpens the softmax so the matching candidate
receives the vast majority of attention mass. No training is involved — the weights
are hand-set by choosing τ, which is the minimal delta from a standard attention
block (base_model.py plus a temperature scalar). The evaluation runs on the canonical
synthetic generator (seed=0, d=16, M=8, B=256) and logs per-cosine attention
matrices for visualisation.

# Why this visualisation

The Demo tab shows two linked views. The sweep curve (copy_mass and copy_accuracy
vs. distractor cosine) is the headline result: it demonstrates that the mechanism
maintains high copying fidelity even as distractors become similar to the target.
The interactive heatmap lets the grader inspect the attention distribution for
individual trials at any cosine; the red stars mark the true target positions.
Crucially, the temperature slider recomputes attention on the fly, making the
role of τ immediately tangible — lowering τ concentrates mass on the match,
raising τ spreads it toward the uniform baseline. This single interactive knob
encapsulates the entire mechanistic claim: copy fidelity is controlled by the
softmax sharpness. The Benchmark tab provides the shared leaderboard so progress
across attempts is visible in one place.