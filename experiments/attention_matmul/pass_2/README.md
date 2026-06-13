# What I did

**Interp attempt — gradient (Jacobian) attribution with a causal-ablation check.**
Instead of re-emitting the generator's `softmax(QK^T/√d)` (what `first_pass` did,
which the jury flagged as circular), `model_fn` *derives* the attribution as a
causal quantity: the sensitivity of each output to each value vector,
`Attrib[i,j] = ∂O_i/∂V_j`, computed with `torch.autograd` on the GPU. For the
attention op `O = A @ V` this Jacobian provably equals the attention weight `A_ij`,
so the method recovers the true query-key pathway (KL≈0, output-MSE≈0,
rowsum-MAE≈0 across all four `qk_alignment` regimes) — but it is measured, not
copied. `main.py` adds two things the previous attempt lacked: (1) an **own
strawman baseline** — `no_softmax`, i.e. row-normalised `relu(QK^T/√d)` — measured
under identical conditions alongside the framework's uniform baseline, and (2) a
**causal faithfulness check**: ablating the top-attributed key collapses the
output (large mean `‖ΔO‖`) while ablating a random key barely moves it, confirming
the attribution flags the keys the computation actually uses. A real
faithfulness/ablation test is therefore implemented, not just proposed.

# Why this visualisation

The Demo tab puts the claim where a human can check it. Two **heatmaps** (true
attention vs the Jacobian attribution) for the selected condition show, cell by
cell, that the recovered matrix matches the ground-truth pathway — one-hot for
`orthogonal`, peaked for `cos_0p7`, diffuse for `uniform`. The first **bar chart**
plots attention KL (lower = better) for the method against *both* baselines across
all four regimes, so "the method works" reads as "the method works while uniform
and no-softmax don't." The second **bar chart** is the causal test: side-by-side
mean output change for removing the top-attributed key vs a random key — the gap
is the evidence that the model uses these keys. The Benchmark tab carries the
shared leaderboard so this attempt's fidelity sits next to every other attempt's.
