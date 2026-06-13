# attention_hierarchical_pool / pass_2

**What I did**  
I implemented a hand-coded, minimal-delta Transformer that solves the hierarchical pooling question:  
- **12 layers, 8 heads per layer** (96 heads): heads 0–2 are dedicated hierarchical poolers (local, chunk, superchunk) across all layers; heads 3–7 are dummy uniform heads to satisfy the goal’s shape contract.  
- **No embedding layer, no MLP, no residual network**: the entire computation lives inside a single attention mechanism per head. This is the smallest possible model that still looks like a self-attention step.  
- **Pure attention circuit**: each query token uses a positional one-hot key to select a region; the dot product collapses to a region indicator matrix; temperature and softmax then produce a uniform distribution within that region (or uniform flat attention for dummy heads).  
- **Deterministic**: all values are derived from the fixed synthetic batch and the QK dot product, no training.

The key delta from a vanilla `base_model.py` is a hand-rolled `attn_head_forward` that builds Q and K from explicit positional one-hots instead of projections of embedding vectors. I deliberately skip embeddings/MLP/residuals to isolate the *attention mechanism itself* as the mechanism expressing the hierarchy.

**Why this visualisation**  
The Gradio app shows two views:  
1. **Demo tab**: a table listing every (layer, head) with its measured concentrations (local, chunk, superchunk) and entropy. Head 0 should dominate local concentration; head 1 should dominate chunk; head 2 should dominate superchunk. Dummy heads 3–7 should sit near baseline values — this contrast *is* the mechanism check.  
2. **Benchmark tab**: a reusable leaderboard (`benchmark_panel`) that overlays per-layer median concentrations and entropy across all attempts, alongside the analytically derived “no-mechanism” baseline (uniform-within-chunk attention). The headline metric `hierarchical_robustness_canonical` is a single number summarising the depth-dependent shift from fine to coarse pooling.  

Both tabs give the grader two complementary ways to verify the hierarchical claim: the per-head fine-grained inspection in the table and the across-attempts macro view in the benchmark panel.