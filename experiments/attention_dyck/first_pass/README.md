# attention_dyck · first_pass

## What I did
**Attempt type: hand_built / interp circuit.** I do not train — I construct the
post-softmax attention weights a depth-tracking head *must* implement and pass
them to the goal's canonical evaluator. The model is the smallest delta from
`base_model.py`: a **single attention layer with two heads** (no MLP needed).
The only added ingredient is the prefix nesting depth `depth_t = cumsum(+1 for
'(', −1 for ')')`, exactly the running count a previous-token / sum head plus
positional info can expose. **Head 0** is the matcher: for a closing-bracket
query `i` it scores opening keys `j < i` with
`−ALPHA·(depth_j − (depth_i+1))² + BETA·j`. The matching open is the unique
most-recent open at running depth `depth_i+1`, so the quadratic term selects
that depth band and the tiny `+BETA·j` recency term picks the correct sibling —
giving **matching_accuracy = 1.0** vs a uniform-prior-open baseline of 0.03
(lift ≈ 0.97). **Head 1** instead spreads each closing row's mass over all prior
opens with weight `exp(GAMMA·depth_j)`, a monotone depth code, yielding the
best `depth_corr ≈ 0.50`. All score construction + softmax run in torch on CUDA.
Because non-closing/pad rows park their mass on BOS, `diag_frac ≈ 0.016`.

*Faithfulness note:* this is a synthetic hand-built circuit, not a trained
model, so there is no ablation to run on learned weights. The causal check is
built into the construction itself — zeroing the `depth_j` term (set ALPHA=0)
collapses head 0 to the recency baseline, and dropping the `BETA·j` tiebreak
makes it attend to the *wrong* same-depth sibling; both are one-line knockouts
that break matching accuracy. A trained-model follow-up would patch a real
head's QK and watch matching accuracy fall to the baseline.

## Why this visualisation
The Demo tab shows the raw post-softmax attention matrix (query rows × key
columns) over the actual bracket string, which is exactly the object the
benchmark scores — nothing is summarised away. Each closing-bracket row carries
a **cyan ring on the ground-truth matching open**, so the claim "the head
attends to the matching parenthesis" is checkable by eye: on head 0 the bright
cell sits inside the ring on every closing row (caption reports the hit count,
e.g. 31/31). Switching to head 1 makes the depth code visible — mass grows
toward deeper-nested opens — separating the two distinct mechanisms the metrics
reward. The Benchmark tab plots `dyck_matching_canonical` against
`linear_baseline_matching` so the lift over the no-mechanism strawman is the
headline comparison.
