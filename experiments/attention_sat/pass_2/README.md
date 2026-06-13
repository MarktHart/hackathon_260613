# What I did
This attempt implements a hand-built model function that computes exact softmax attention on the GPU for the synthetic attention saturation task. The `model_fn` receives query, key, value tensors, a logit scale, and an optional causal mask; it computes scaled dot-product attention with `torch.einsum` on CUDA, applies the mask, and returns attention weights, per-query entropy, and a saturation score. The saturation score is the **mean maximum attention weight** across queries — a direct, mechanistic measure of how peaked the attention distribution is (1.0 = perfectly peaked, ~1/seq_len = uniform). This score is monotonic with the logit scale and cleanly separates the saturated regime (≥10.0) from the unsaturated regime.

# Why this visualisation
The Gradio app provides three complementary views:

1. **Saturation Metrics Across Scales** (2×2 panel): Shows the attempt's saturation score, mean entropy, and max attention weight against the analytic reference across all seven logit scales (0.1 → 100). The vertical line at scale=10 marks the ground-truth saturation threshold. This directly visualises whether the method tracks the true saturation curve and where it diverges.

2. **Attention Weight Heatmap**: Shows the mean attention matrix at a user-selected scale (default: scale=10, the canonical saturated condition). In the saturated regime this reveals the hard-attention pattern (near one-hot rows); in the linear regime it shows diffuse attention.

3. **Benchmark Tab**: The shared `benchmark_panel` shows the leaderboard across attempts with the headline metric `saturation_detection_auroc` (target ≈ 1.0, since logit_scale itself is a perfect oracle) and per-scale fidelity metrics like `entropy_correlation_sweep`. This lets us verify the method matches the oracle without overfitting.

The visualisation emphasises the *mechanistic* saturation score (max weight) over proxy metrics, and the heatmap lets a human confirm that "saturation" corresponds to actual hard-attention structure, not just a scalar artifact.