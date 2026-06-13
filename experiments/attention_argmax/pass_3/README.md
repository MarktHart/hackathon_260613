# What I did

**Attempt type: hand_built (analytic circuit, no training).** I express the
attention head as a single line of scaled dot-product softmax — `softmax(K·q /
tau)` — written as torch tensors on CUDA, with hand-set temperature `tau = 0.25`
and no learned parameters. This is the smallest possible delta from
`base_model.py`'s attention block: one head, no MLP, no residual stack — just
the softmax-over-similarities that the base model already contains, with a
temperature knob. The claim it tests is that a softmax head *is* a soft argmax:
as `tau → 0` the distribution converges to a one-hot on the highest-similarity
key. On the canonical sweep this gives ~0.99 winner mass at `separation = 2.0`,
near-zero entropy, and rank 1. To show the `exp` is load-bearing (faithfulness /
baseline), `main.py` also runs a noise sweep contrasting this head against a
no-`exp` strawman (`relu` normalisation) and the uniform `1/N` baseline. Because
this is a purely synthetic circuit there is no trained model to ablate; the
exp-vs-no-exp swap is the causal knockout — removing the non-linearity destroys
the argmax behaviour while the rest of the pipeline is held fixed.

# Why this visualisation

The Demo tab puts the two heads **side by side** as bar charts over the 32 key
positions, with the true winner in red and the uniform line dashed. This is the
smallest artefact that, if flipped, changes the claim: you can see the softmax
head spike on the red bar while the no-`exp` head smears mass across noisy
distractors. The sliders let the grader drive `separation`, `noise`, and `tau`
and watch the spike sharpen or melt in real time — directly exercising the
`tau → 0 ⇒ argmax` story. Below it, the noise-sweep line plot puts *winner mass*
on the y-axis against key-noise std on a **log x-axis spanning >2 orders of
magnitude**, with the uniform baseline as the floor — showing exactly where the
argmax property degrades and that the no-`exp` baseline collapses to uniform.
The Benchmark tab carries the cross-attempt leaderboard so iteration shows up as
a curve on the headline `argmax_fidelity_canonical` metric.
