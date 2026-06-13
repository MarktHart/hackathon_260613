import torch
import numpy as np
from agentic.experiments import load_task, record_benchmark, results_dir

# GPU requirement
DEVICE = "cuda"

# Hand-built model function that performs attention-like computation
def model_fn(seq: tuple[int, ...], i: int, j: int) -> np.ndarray:
    # Convert sequence to tensor on GPU
    tensor_seq = torch.as_tensor(seq, dtype=torch.long, device=DEVICE)
    
    # Simple attention mechanism implementation
    # Creates scores based on bracket depth and proximity to potential split points
    n = len(seq)
    
    # Initialize score tensor on GPU
    score_tensor = torch.zeros(n + 1, dtype=torch.float32, device=DEVICE)
    
    # For each possible split point, calculate attention score
    for k in range(i + 1, j):
        # Score based on bracket depth
        bracket_depth = 0
        for pos in range(i, k):
            if seq[pos] == 0:  # '('
                bracket_depth += 1
            else:  # ')'
                bracket_depth -= 1
        
        # Score based on balance point detection
        if bracket_depth == 0:  # Balance point
            score_tensor[k] += 1.5  # Higher weight for balance points
        else:
            score_tensor[k] += 0.75  # Moderate weight for other split points
    
    # Convert back to NumPy and return
    return score_tensor.cpu().numpy()

# Load task and evaluate
task = load_task(__file__)
payload = task.evaluate(model_fn)

# Record benchmark results
record_benchmark(__file__, results_dir(__file__), payload)