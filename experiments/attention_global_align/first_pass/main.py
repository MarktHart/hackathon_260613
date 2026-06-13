import torch
import numpy as np

DEVICE = "cuda"

def model_fn(q: np.ndarray, K: np.ndarray) -> np.ndarray:
    # Convert to CUDA tensors
    qt = torch.as_tensor(q, dtype=torch.float32, device=DEVICE)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=DEVICE)
    
    # Compute K @ q with torch (GPU)
    logits = Kt @ qt
    
    # Return NumPy array
    return logits.detach().cpu().numpy()

if __name__ == "__main__":
    from agentic.experiments import load_task, record_benchmark, results_dir
    
    task = load_task(__file__)
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, results_dir(__file__), payload)