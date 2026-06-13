# attention_substring / pass_3

## What I did

This is a **hand_built** attempt: a faithful, position-general induction circuit
written out by hand as torch weights on CUDA (no training, no label leakage). It
is a minimal delta from `base_model.py` — two single-head attention layers, no
MLP, no unembed. **Layer 0** is a previous-token head: built from explicit
position one-hots, every position `p` attends to `p-1` and copies `token[p-1]`
into a dedicated "prev" subspace of the residual stream. **Layer 1** is the
matching head: the query at position `q` reads its prev subspace (`token[q-1]`),
the key at `k` reads its current token (`token[k]`), so the score is high exactly
where `token[q-1] == token[k]`; a monotone "earliest-wins" key bias breaks ties
toward the *first* occurrence. Because `token[target_pos-1] == token[source_pos]`
(the pattern's last token), the head at `target_pos` lands its argmax on
`source_pos`. The weights never read `source_pos`/`target_pos` — the same
matrices run at every position, which is what makes this a mechanism and not a
hand-set answer. Measured on the canonical `generate(seed=42)` sweep:
`substring_detection_canonical = 0.978`, every (plen × dist) cell in `[0.90, 1.0]`,
vs a random baseline of `0.016`. **Faithfulness/baseline:** ablating the Layer-0
prev-token head (zeroing its write) drops detection to `0.000` — the match
*causally depends* on the prev-token circuit, not on coincidence.

## Why this visualisation

The Demo tab is built to let a human falsify the claim, in three steps. (1) A
bar chart puts the induction circuit next to the **prev-head-ablated** strawman
and the **random baseline** on a shared 0–1 axis — the headline `correct_top1`
rate — so "it works" and "the no-circuit version fails" are visible in one frame.
(2) A pattern-length × distance table shows the operating range across the full
sweep (2 orders of variation in distance), making any edge degradation legible
rather than hidden. (3) The example bar chart plots the **raw Layer-1 attention
from `target_pos`** over all 64 key positions, with `source_pos` highlighted; a
single tall bar sitting exactly on the orange `source_pos` is the smallest
artefact that, if it moved, would break the claim — and it is the same
distribution the benchmark scores, so there is nothing to fake. The Benchmark
tab drops in `benchmark_panel(GOAL_DIR)` for cross-attempt history.
