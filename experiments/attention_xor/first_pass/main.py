import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

DEVICE = "cuda"


def model_fn(tokens: np.ndarray) -> np.ndarray:
    """
    Hand-built model that implements an XOR logit via an affine combination
    of the two binary features (A and B) plus their product (A*B) and a bias.

    Args:
        tokens: (N, 4) int32 tensor with columns [CLS, A_tok, B_tok, SEP].
            A_tok in {1,2} -> A in {0,1}; B_tok in {3,4} -> B in {0,1}.

    Returns:
        logits: (N,) float32 logits, predict XOR=1 if logit > 0.
    """
    tok = torch.as_tensor(tokens, device=DEVICE)
    # Extract features
    A = (tok[:, 1] == 2).to(torch.float32)  # 2 => A=1, 1 => A=0
    B = (tok[:, 2] == 4).to(torch.float32)  # 4 => B=1, 3 => B=0

    # XOR is affine in (A, B, AB): y = 2A + 2B - 4AB - 1
    # logit = 2*(A XOR B) - 1, i.e. +1 when XOR=1 and -1 when XOR=0.
    feature = torch.stack([A, B, A * B], dim=-1)   # (N, 3)
    W = torch.tensor([2.0, 2.0, -4.0], device=DEVICE)  # weight coeffs
    b = -1.0                                            # bias term
    return (feature @ W + b).detach().cpu().numpy()


payload = task.evaluate(model_fn)

run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)
print(f"Benchmark written to {run_dir / 'benchmark.json'}")
