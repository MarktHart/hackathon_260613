# What I did

**Attempt type: hand_built (every weight set by hand, no training).** I took the
single attention head from `base_model.py` and reduced it to the smallest circuit
that expresses palindrome-ness as *positional alignment*. The delta from
`base_model.py`: drop the MLP, RoPE and learned embeddings; make Q and K depend
only on **position** so that query `i` attends to key `L-1-i` (a one-hot dot
product at temperature 30 → a hard anti-diagonal routing after softmax); let the
value carry the **token one-hot**. The attention output at `i` is therefore the
one-hot of the *mirror* token, and a dot product with the token one-hot at `i`
returns 1 iff `token_i == token_{L-1-i}`. The palindrome score is the count of
agreeing positions (`SEQ_LEN − 2k` for a `k`-broken negative), which is strictly
monotone in `k` — giving rank-AUC **1.0 at every slice including the diagnostic
`k=1`**, and headline `palindrome_robustness = 1.0`, while the histogram baseline
sits at ~0.52. All compute is torch tensors on CUDA. For faithfulness I re-run the
exact same circuit under two ablated routings (attend-to-self `identity`, and an
off-by-one `shift`); both collapse to chance, showing the mirror routing is the
load-bearing part rather than the embeddings or the readout.

# Why this visualisation

Two panels carry the whole claim. The **attention heatmap** shows the routing
*directly*: a clean anti-diagonal means position `i` really does send its mass to
`L-1-i` — this is the mechanism, not a description of it. The **AUC-by-difficulty
bar chart** puts the right thing on the y-axis (rank-AUC, the exact benchmark
metric) against the right x-axis (broken-pair count `k`, hardest at `k=1`), and
places three bars side by side at each `k`: the mirror head, the histogram
baseline, and the identity-routing **ablation**. A reader sees in one glance that
the green bars stay pinned at 1.0 across two-plus orders of difficulty while both
the strawman and the ablated circuit hug the 0.5 chance line — i.e. the mechanism
both *works* and is *necessary*. The worked-example text grounds the abstraction
with a real palindrome and a one-pair-broken negative and their scores.
