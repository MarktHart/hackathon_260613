# What I did

This is a **hand_built** single attention head — `base_model.py`'s scaled-dot-product
attention with exactly one change: the softmax scaling. Given the query `Q` and the `K`
unit keys, the head returns `softmax(β · (K @ Q))` where `β` is a **hand-set, Bayes-optimal
inverse temperature** derived from the generative noise model, not learned. Because the
generator builds `Q = K_target + n` with `n ~ N(0, σ²I)`, `σ² = 1/(d·10^(SNR/10))`, the
posterior over which key is the target is `∝ exp(‖Q'‖·(Q·K_i)/σ²)`, so the optimal head is
plain attention with `β = ‖Q'‖/σ² ≈ 671` (vs the vanilla `1/√d ≈ 0.125`, ~5000× too small).
With that one constant, target attention is ≈1.0 across the whole sweep (ρ = 0.25→4.0,
K = 16→256), giving `scc_auc ≈ 1.0` versus the vanilla head's ≈chance — so the discovery
is that **temperature, not the geometry, was the capacity bottleneck**. `main.py` also runs
an **ablation** (knock out the `exp` → relu/sum; knock out the temperature → vanilla 1/√d;
knock out the query → uniform) showing every piece is causally necessary, and records the
target-minus-best-distractor logit gap, which stays positive at every ρ (the target is the
argmax ~100% of the time, which is *why* a sharp softmax wins). This attempt is purely
synthetic (no trained model), so the "ablation" is a circuit knock-out on the hand-built
head; the faithful causal test on a real trained head would be activation-patching the
head's softmax temperature/scale and watching target attention collapse to chance — exactly
the red curve here.

# Why this visualisation

The headline plot puts **target attention mass on the y-axis against ρ = K/d on the x-axis**
— the exact quantity the goal asks for — and overlays the failing vanilla `1/√d` head (red)
and the `1/K` chance line (gray) so the "works vs fails" contrast is one glance, on the same
head and same data. The **temperature sweep** is the load-bearing panel: it plots capacity
(`scc_auc`) against the logit multiplier on a log axis with markers on the vanilla scale and
the Bayes-optimal β, making it visually obvious that the only thing separating chance from
perfect is where you sit on the temperature axis. The **ablation bar chart** shows each
knock-out dropping to chance, evidence the mechanism is exactly exp-sharpened, temperature-
calibrated, query-conditioned attention. The **logit-gap panel** explains the mechanism
geometrically: the target's dot-product stays well above the best distractor at every ρ, so
the head *can* resolve all K features — capacity was never the limit.
