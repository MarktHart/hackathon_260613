# attention_xor pass_5 — hand-built attention + ReLU-MLP XOR circuit

**What I did**
This is a **hand_built** attempt (no training), kept as a minimal delta from
`base_model.py`: a single attention head feeding a two-unit ReLU MLP, with every
weight set by hand and all compute on CUDA. The mechanism factors XOR into two
faithful transformer steps. **(1) Attention pools the bits into a sum:** the CLS
token's query points at the "A-type" and "B-type" key channels, so softmax puts
≈0.5 weight on each of `A_tok` and `B_tok`, and the pooled value channel reads
out `s = A + B ∈ {0,1,2}`. **(2) A ReLU bump selects `s == 1`:**
`logit = 0.5 − relu(s−1) − relu(1−s)`, which is `+0.5` exactly when `s=1`
(XOR=1) and `−0.5` at `s=0,2` (XOR=0). The two ReLUs are the non-linearity the
best linear probe provably cannot express, so the circuit captures the full
above-baseline headroom (`xor_robustness ≈ 1.0`) at every marginal. A built-in
**ablation** (`ablation_fn`) keeps the same attention but removes one ReLU,
degrading the bump into a monotone NAND that is linearly separable and therefore
collapses back onto the linear floor — direct evidence the XOR behaviour lives
in the ReLU MLP, not the pooling.

**Why this visualisation**
The Demo tab makes both steps checkable on a single example: it shows the CLS
attention weights (≈0.5/0.5 on the two feature tokens), the pooled sum `s`, and
the resulting bump logit, alongside the full four-cell truth table. The bar
chart puts the **circuit** logit next to the **ablation** logit for each input
cell on a shared `[-1.1, 1.1]` axis — the circuit is positive on exactly the two
XOR=1 cells while the ablation is positive on three cells, so a glance shows the
second ReLU is what carves out the non-separable band. The Benchmark tab drops in
the shared `benchmark_panel` so the lift over the best-linear-probe floor and the
headline `xor_robustness` are comparable across every attempt and every marginal.
