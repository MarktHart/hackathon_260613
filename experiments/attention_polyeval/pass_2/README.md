# attention_polyeval — pass_2 (hand_built)

## What I did
This is a **hand_built** attempt: a single attention block that evaluates the
quadratic **x²** elementwise, with no MLP and no second layer. It is
`base_model.py`'s attention specialised so that every token attends to **itself**
and to one learned constant **sink** key. The self-attention score is the diagonal
QK bilinear form **β·x_f²** (set Q = K = √β·x, so Q·K = β·x²) — this is where the
squaring physically happens, inside the attention score. The 2-key **softmax**
(self vs. sink) is the only nonlinearity; it turns β·x² into a bounded weight
`p = σ(β·x² − b₀)` that is monotone in x², and the output projection W_O is a
hand-set affine readout `α·p + γ` calibrated (offline, from the input
distribution only — never the targets) to match x². On the canonical degree-2
task this reaches R² ≈ 0.99 versus the linear baseline's R² ≈ 0, so
`poly_eval_headline ≈ +0.99`. Two faithfulness ablations are saved: replacing the
quadratic QK score with a **linear** one collapses R² to the linear-baseline floor
(`ablation.json`), and a **scale sweep** over 0.03→30 (`scale_sweep.json`) shows
the mechanism holds across >3 orders of magnitude when the QK gain β tracks the
input scale, and where it saturates when β is held fixed. Unlike the previous
pass, the squaring is a real, traceable QK·softmax operation rather than random
weights summed into a constant.

## Why this visualisation
The Demo tab leads with the claim itself: a scatter of the block's output against
the input with the **x² parabola** overlaid — if the attention genuinely squares,
the points lie on the parabola, which a human can verify at a glance. The second
panel is the load-bearing **causal** evidence: a three-bar R² chart (mechanism vs.
linear-QK ablation vs. linear baseline) on the degree-2 target — knocking out the
quadratic QK term drops the green bar to the grey floor, proving the QK square is
what does the work rather than the readout. The third panel puts R² on the y-axis
against a log input-scale x-axis to show the operating range and exactly where the
fixed-β variant breaks. The Benchmark tab carries the cross-attempt leaderboard
and history so iteration shows up as a curve.
