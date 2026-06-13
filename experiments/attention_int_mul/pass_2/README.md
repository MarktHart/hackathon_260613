# What I did

This is a **trained** attempt (with a hand-built companion and an ablation), and
it answers the goal's question directly: *can one attention head multiply?* The
mechanism is the smallest delta from `base_model.py`'s `Attention` — instead of
forming one query per token via the linear `qkv` and scoring keys by an
*additive* `q·k`, the head treats the two operands as two query tokens `φ(a)`,
`φ(b)` and forms its scoring query by a **bilinear / Hadamard-product
interaction**: `q = W_o((W_a φ(a)) ⊙ (W_b φ(b)))`, then `logit_i = β·⟨key_i, q⟩`.
The elementwise product is the multiplication — it makes `q` a genuine quadratic
function of the operands, which no additive head can express. I train
`W_a, W_b, W_o, β` with a routing cross-entropy on the task's own trial
distribution (training seeds 1–30, validation 101–105, **never the eval seed
42**); it reaches **1.00 mean routing accuracy** and **1.00 attended mass** at
the canonical K. Two controls run through the identical `task.evaluate` harness:
(a) a **hand-built** `(d,d,d)` bilinear tensor built from the exposed table via
dual operand bases — a true multiplicative circuit (not argmax-decode + lookup,
unlike the first pass), also 1.00; and (b) the **faithfulness ablation** — the
*same trained weights* with `⊙` swapped for `+`, which collapses routing to
**0.14**, *below* the additive baseline (0.32). That swap is the causal test: the
multiplication, not the linear projections, is what routes to `a·b`.

# Why this visualisation

The Demo's main panel puts **routing accuracy on the y-axis against operand range
K on the x-axis** — exactly the goal's headline metric and its hardest axis — and
places four bars side by side per K: the trained multiplicative head, the
hand-built circuit, the additive baseline, and the ablated (`⊙→+`) head, with a
chance line. This makes the claim falsifiable in one glance: multiplication stays
near 1.0 across two orders of magnitude of K while both additive variants sit at
or below baseline. The second panel is a real scatter of **attended integer value
vs. true product `a·b`** at the canonical K (replacing the prior pass's empty
placeholder): every point on the `y=x` diagonal is a trial routed to the product,
so misses would jump off the line. The training curve confirms the head *learned*
the mechanism rather than having it imposed. The Benchmark tab drops in the shared
panel so pass_2 ranks against first_pass and any future attempts.
