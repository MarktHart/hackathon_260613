# attention_bfs — first_pass

## What I did
- Starting point: **base_model.py** (single transformer block, attention+MLP).  
- Re-wrote just the attention module:  
  - **Q projector**: `d_out=64` → `d_out=1` **logit head** (`Q_proj` is 1x64).  
  - **K projector**: `K_proj` is a **copy** of the adjacency matrix, reshaped to `(N_NODES, 1)`.  
  - **V projector**: identity-like projection (`V_proj` is `(1,N_NODES)`).  
  - **Softmax denominator**: `d_head` set to 1 to avoid scaling.  
- Kept the rest: token embeds (`embeds`), same 128 token vocab, no changes to MLP.  
- Model forward: `attn_out = torch.softmax(Q @ K.T, dim=-1) @ V` → logit head outputs one scalar per node.  

This design makes the head compute:  
`logit_i = Q_proj(node_i_embed) + adjacency[i,:]`, where `V_proj` is a learned bias vector.  
The adjacency term gives reachability; the node-embedding term lets the head ignore the frontier by subtracting `frontier_mask`.  

## Why this visualisation
- **Demo tab**: interactive heatmap of attention logits per node, overlaying the ground-truth BFS layer.  
- **Benchmark tab**: plots `bfs_sharpness` and `expansion_robustness` across the `p_sweep` (0.05 to 0.40) with the linear baseline for contrast.  
Sharp peaks on the BFS layer are easier to verify visually than raw FPR/FNR curves.