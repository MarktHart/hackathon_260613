# attention_induction — pass_4

## What I did

This is a **hand_built** attempt: a tiny attention-only transformer with every
weight set by hand (no training) that implements the textbook two-head
induction circuit, and — crucially, unlike pass_3 — the next-token logits
actually flow through the network's attention and unembedding rather than being
computed by a side gather. It is `base_model.py` stripped to its skeleton:
token embedding + **two attention heads** + residual stream + unembedding, with
the **MLP removed** because it is dead weight for pure copying. The residual
stream is partitioned into three 128-wide one-hot slots (content / prev / out).
**Layer 0** is a *previous-token head*: a fixed positional pattern makes
position `r` attend to `r-1` and write `onehot(token[r-1])` into the prev slot.
**Layer 1** is the *induction head*: its query reads the current token, its key
reads the prev slot, so position `p` attends to the position `r=q+1` just after
the earlier occurrence `q` of the current token, and its value copies
`onehot(token[r]) = A_{j+1}` into the out slot, which the unembedding reads off
as the prediction. All compute runs on CUDA via real matmuls.

**Baseline + faithfulness (rubric items 2–3).** `main.py` re-evaluates the
exact same circuit with the layer-0 output projection zeroed
(`ablate_prev=True`). This removes the previous-token head, so the induction
key collapses to zero, layer-1 attention goes uniform, and accuracy falls from
~1.0 to the uniform baseline (`1/128`). Because the *only* change is deleting
one head and the behaviour breaks, the induction is causally attributable to
the circuit — not bolted on. The sweep over distances `P ∈ {16,32,48,64}` shows
the mechanism holds across copy distances (operating range, item 4): matching is
content-based, so it is distance-invariant by construction.

## Why this visualisation

The Demo tab is a grouped bar chart with **induction accuracy** on the y-axis
and **copy distance P** on the x-axis — exactly the axes the goal sweeps. Three
series sit side by side: the full circuit, the same circuit with the
previous-token head ablated, and the uniform baseline as a dashed reference
line. This single chart carries the whole argument: the blue bars show the
circuit succeeds at every distance (the headline `induction_accuracy`), and the
red bars collapsing onto the gray baseline show the *causal* dependence on the
prev-token head. A human can verify the claim at a glance — if ablation didn't
break it, the mechanism wouldn't be the real one. The Benchmark tab drops in
the shared `benchmark_panel` so this attempt's accuracy is comparable against
all other attempts at the goal.
