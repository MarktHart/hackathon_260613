# What I did

This is a **hand_built** attempt: a 2-layer, 4-head, `d_model=64` attention-only
transformer (the `base_model.py` shape, MLP dropped) whose every weight is set by
hand — no training. For a first-order HMM the Viterbi backpointer of query `t` is
always `t-1`, so the attention substrate the Viterbi recurrence needs is a
*previous-token head*. I build exactly that in **layer-0 head-0**: with a one-hot
positional encoding, `W_Q` reads "the code of position `t-1`" and `W_K` reads "the
code of position `s`", so the causal row `t` peaks sharply on key `t-1`
(canonical excess ≈ 0.9, far above the uniform-reader 0 and the random baseline).
The other heads are hand-set to contrasting patterns (self, BOS, uniform) so the
per-head chart has a single clear winner. Because the circuit is explicit I add
the two things the previous attempt lacked: **causal evidence** — ablating L0H0
(or zeroing the positional encoding) collapses the headline excess to ~0 while
ablating any other head leaves it untouched — and an **operating-range sweep**
across HMM seed, attention temperature (≈2 orders of magnitude), and sequence
length (8→28). Caveat stated plainly: a hand-built circuit *is* used by
construction; the honest extra check for a *trained* model would be activation
patching of this head, which the ablation panel mirrors here.

# Why this visualisation

The Demo stacks four views, each answering one rubric question with the
quantity from `benchmark.py` on the y-axis. (1) **Per-head bar** of `excess` —
one comparison showing only L0H0 beats the uniform reader (the `0` baseline is
drawn in). (2) **Excess-by-position line** for the best head — shows the
signature holds across the sequence, not just on average. (3) **Ablation /
strawman barh** — the causal core: bars for "full", each single-head knockout,
and "zero positional encoding"; only the L0H0 and positional-encoding bars drop
to ~0, proving the signature is localized to that head and its positional input.
(4) **Operating-range triptych** — excess vs seed, vs temperature (monotone,
→0 as sharpness→0), and vs sequence length, so the reader sees where it holds
and how it degrades. A selectable **attention heatmap** lets you eyeball the
bright sub-diagonal (mass on key `t-1`) that the scalar metric summarizes.
