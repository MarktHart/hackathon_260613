import torch
import numpy as np
from agentic.experiments import load_task, record_benchmark, results_dir

def main():
    task = load_task(__file__)
    run_dir = results_dir(__file__)

    DEVICE = "cuda"
    query_tensor = torch.tensor(np.zeros(32, dtype=np.float32), device=DEVICE)
    keys_tensor = torch.tensor(np.zeros((3, 32), dtype=np.float32), device=DEVICE)

    def model_fn(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        qt = torch.as_tensor(query, dtype=torch.float32, device=DEVICE)
        kt = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)
        scores = qt @ kt.t()
        attn_weights = torch.softmax(scores, dim=0).detach().cpu().numpy()
        return attn_weights

    payload = task.evaluate(model_fn)
    record_benchmark(__file__, run_dir, payload)

if __name__ == "__main__":
    main()