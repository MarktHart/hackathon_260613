# What I did

This is a **hand-built (interp) circuit** whose central claim is that carry
propagation — the goal's hard part — is a genuine **attention mechanism**, not a
loop. pass_2 scored well but the jury's top complaint was *architecture fit*: its
carries were an explicit Python `for`-loop ripple. Here the carry chain is a
single **carry-lookahead softmax attention** layer (`carry_lookahead_attention`):
for each answer column `i`, one attention head queries the lower columns and
lands one-hot on the *nearest decisive* column `j* = max{ j<i : s_j ≠ 9 }`,
reading its generate signal `g_j = [s_j ≥ 10]`. Columns with digit-sum 9 are
"propagators" given a large negative score bias, so attention skips an entire run
of 9s — `999+1` makes the THOUSANDS query attend two columns down to the units
column, the full carry chain expressed as one parallel attention with no
iteration over columns. The model is `base_model.py` plus a value embedding
(id→digit), a hand-set digit-routing attention layer (the easy column fetch), and
this hand-set carry-lookahead attention layer; all torch on CUDA, nothing
trained. Faithfulness is built in and causal: **ablating the carry-attention
layer** (carries forced to 0) collapses the circuit *exactly* to the task's
linear baseline, so exact-match dies on every carrying slice while the full
circuit stays at 1.0 (`carry_robustness = 1.0`) across 8 held-out seeds; and
because the mechanism is digit-width agnostic, it stays exact on 3→12-digit
operands (carry chains 4× the canonical length, operands to 10¹²).

# Why this visualisation

The Demo's headline artefact is the **carry-lookahead attention heatmap**: rows
are queries (carry into each column, incl. the leading digit), columns are source
columns, cell = attention weight. This is the smallest object that, if it routed
differently, would break the claim — on `999+1` you watch the bright cell in the
thousands row sit on the *units* column, two propagator-9s skipped, making
"carry = attention" literally visible rather than asserted. Beneath it, the
**faithfulness bars** put exact-match (y) against carry count (x) for full vs
carry-attention-ablated: same circuit, one attention layer removed, and the
carrying slices flip from 1.0 to 0 — the minimal causal flip the goal asks for.
The **operating-range line** puts exact-match (y) against operand digit width (x,
i.e. carry-chain length) for the lookahead vs the linear baseline, showing the
attention is the adder across two orders of magnitude of operand scale, not a
3-digit lookup table.
