import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

def model_fn(pattern: np.ndarray, embed: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """
    Hand-built attention mechanism for regex-like pattern matching.
    
    For each position i in the residual stream, compute how well the
    preceding L tokens (where L = len(pattern)) match the pattern.
    Non-wildcard pattern positions contribute dot-product similarity
    between the residual at the corresponding offset and the pattern
    token's embedding. Wildcards contribute zero (match anything).
    """
    # Convert to torch on GPU
    pattern_t = torch.as_tensor(pattern, dtype=torch.long, device=DEVICE)
    embed_t = torch.as_tensor(embed, dtype=torch.float32, device=DEVICE)
    residual_t = torch.as_tensor(residual, dtype=torch.float32, device=DEVICE)
    
    N, d = residual_t.shape
    L = pattern_t.shape[0]
    
    # logits output
    logits = torch.zeros(N, dtype=torch.float32, device=DEVICE)
    
    # Mask of non-wildcard positions in pattern
    mask = (pattern_t != -1)
    if not mask.any():
        # All wildcards: uniform attention
        return logits.detach().cpu().numpy()
    
    # Indices of concrete tokens in pattern
    concrete_idx = torch.where(mask)[0]           # shape (K,)
    concrete_tokens = pattern_t[concrete_idx]     # shape (K,)
    concrete_embeds = embed_t[concrete_tokens]    # shape (K, d)
    
    # For each concrete pattern position j, we want to compare
    # residual[i - L + 1 + j] with embed[pattern[j]] for all valid i.
    # Valid i range: L-1 to N-1 (window fits in sequence)
    # For a given j, the residual index is i - L + 1 + j = i - (L - 1 - j)
    # This is a shifted version of residual.
    
    # We'll accumulate scores for each position i
    # Initialize scores for valid positions only
    scores = torch.zeros(N, dtype=torch.float32, device=DEVICE)
    
    for k, j in enumerate(concrete_idx):
        # j is the pattern position (0-indexed)
        # The residual position that aligns with pattern[j] when match ends at i is:
        # res_pos = i - L + 1 + j
        # So i = res_pos + L - 1 - j
        # For each res_pos in [0, N-1], the match-end position is i = res_pos + L - 1 - j
        # But we only care about i in [L-1, N-1], so res_pos in [j, N - L + j]
        
        shift = L - 1 - j  # how much to shift residual to align with match-end positions
        # residual[res_pos] contributes to logits[res_pos + shift]
        # res_pos ranges from j to N - L + j (inclusive)
        # So valid match-end positions i range from L-1 to N-1
        
        # Get the relevant slice of residual
        start_res = j
        end_res = N - L + j + 1  # exclusive
        if start_res < end_res:
            residual_slice = residual_t[start_res:end_res]  # shape (window_len, d)
            # Target embedding for this pattern position
            target_embed = concrete_embeds[k]  # shape (d,)
            # Dot product similarities
            sims = residual_slice @ target_embed  # shape (window_len,)
            # Accumulate into logits at corresponding match-end positions
            logits_start = start_res + shift
            logits_end = logits_start + sims.shape[0]
            logits[logits_start:logits_end] += sims
    
    # Normalize by number of concrete tokens (average similarity)
    num_concrete = concrete_idx.shape[0]
    logits = logits / num_concrete
    
    return logits.detach().cpu().numpy()


def main():
    task = load_task(__file__)
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    print(f"Benchmark recorded to {run_dir / 'benchmark.json'}")


if __name__ == "__main__":
    main()