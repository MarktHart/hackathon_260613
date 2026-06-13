## What I did

**Attempt type: hand_built (a genuine QK attention circuit, not an answer-key
lookup).** This is a single attention layer in the spirit of `base_model.py`
with the MLP dropped — one head's worth of mechanism, replicated across the 4
heads. Each node gets a structural embedding `[tree-address one-hot blocks ;
depth one-hot]`, where the *address* is the path of left/right turns from the
root (left=0, right=1) — exactly the kind of positional feature a transformer
keeps in its residual stream. The key projection `Wk` emits each node's **own**
address+depth code; the query projection `Wq` emits the **address code of the
node it wants to attend to** (parent / grandparent / leftmost-descendant /
next-sibling, selected by `path_rule`) plus a `W`-weighted desired-depth term.
Attention is then `softmax(T · Q Kᵀ)` with self masked out: the dot product
peaks only for the unique node whose own address equals the query's desired
address *and* whose depth matches — i.e. the true target. The target's
**position is resolved by the dot product**, never indexed directly (the key
difference from the rejected `first_pass`, which wrote a one-hot at the
ground-truth position). `Wq`/`Wk` are fixed, **depth-independent** selection
matrices, so the same circuit traces parents in depth-2, depth-3 and depth-4
trees (~0.999 correct attention everywhere via a real softmax, not a hardcoded
1.0). `main.py` adds two causal ablations on the identical pipeline: zeroing the
address block of `K` (`addr_ablated`) leaves the head only able to match depth,
and permuting the codes across positions (`scrambled`) collapses it to the
uniform baseline — proving the head *uses* the address geometry. A synthetic
attempt has no learned model to patch; the honest causal check here is these
two ablations of the circuit's inputs, and a trained version would have to
*learn* both the address features and this QK lookup.

## Why this visualisation

The Demo tab puts the mechanism on screen so the claim is checkable without the
README. The **heatmap** (query rows × key cols, color = attention) with a red ✕
on each query's ground-truth target lets you see the bright cell sit exactly on
the target for every row — the whole point of "tracing the path." The **query
bar** zooms one row out and colors the target bar red, so you read the actual
weight on the correct node (~0.999) versus the smear of near-misses, confirming
it's a sharp-but-real softmax. The **ablation bar** is the faithfulness panel:
full circuit ≈1.0, address-ablated ≈¼ (depth-only), scrambled ≈0.07 baseline —
if the head didn't rely on the address codes these bars would be flat. The
**sweep bar** (depths 2/3/4 and four path rules, with the 1/14 uniform baseline
dashed) shows operating range across tree sizes and relation types in one
comparison. The Benchmark tab drops in the shared leaderboard so this hand-built
ceiling can be compared against future trained attempts.
