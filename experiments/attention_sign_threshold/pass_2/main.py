import torch
import torch.nn as nn
import numpy as np
from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

DEVICE = "cuda"

# A tiny self-attention-style head with a hand-coded sign-thresholding readout:
# the per-pair dot product is squashed by a sigmoid whose sharpness (temp) and
# threshold are learnable, producing a sharp sign flip at dot(q, k) = 0.


class SignThresholdHead(nn.Module):
    def __init__(self, d_model: int, temp: float = 1.0):
        super().__init__()
        self.q_head = nn.Linear(d_model, d_model, bias=False)
        self.k_head = nn.Linear(d_model, d_model, bias=False)
        self.v_head = nn.Linear(d_model, d_model, bias=False)   # unused for this task
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Sharpness hyperparameter: larger = sharper sign transition
        self.temp = nn.Parameter(torch.tensor(temp))

        # Bias that shifts the logistic threshold away from 0.0 if needed
        self.threshold = nn.Parameter(torch.zeros(1))

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        queries: (n_pairs, d_model)
        keys:    (n_pairs, d_model)
        returns: (n_pairs,) per-pair attention weight in [0, 1]
        """
        q = self.q_head(queries)          # (B, d_model)
        k = self.k_head(keys)             # (B, d_model)
        dot = torch.einsum("bd,bd->b", q, k)                     # (B,)
        return torch.sigmoid(self.temp * (dot - self.threshold))  # (B,)


def train_head(head: SignThresholdHead, queries: torch.Tensor, keys: torch.Tensor,
               cosines: torch.Tensor, n_steps: int = 500, lr: float = 1e-3):
    """Push the sign flip toward dot(q, k) = 0 using the canonical sweep.

    Target weight is 1 where the cosine is positive, 0 where negative.
    """
    head = head.to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    targets = (cosines > 0).float()

    for step in range(n_steps):
        optimizer.zero_grad()
        scores = head(queries, keys)            # (n_pairs,)
        loss = ((scores - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"step {step:04d} loss={loss.item():.4f} "
                  f"temp={head.temp.item():.2f} threshold={head.threshold.item():.2f}")

    head.eval()
    for p in head.parameters():
        p.requires_grad = False
    return head


# Load the canonical sweep data and move it to the GPU.
_batch = task.generate()
_queries = torch.as_tensor(_batch.queries, dtype=torch.float32, device=DEVICE)
_keys = torch.as_tensor(_batch.keys, dtype=torch.float32, device=DEVICE)
_cosines = torch.as_tensor(_batch.cosines, dtype=torch.float32, device=DEVICE)

head = train_head(SignThresholdHead(d_model=64, temp=3.0).to(DEVICE),
                  _queries, _keys, _cosines, n_steps=500)


def model_fn(queries_np: np.ndarray, keys_np: np.ndarray) -> np.ndarray:
    q = torch.as_tensor(queries_np, dtype=torch.float32, device=DEVICE)
    k = torch.as_tensor(keys_np, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        scores = head(q, k)  # (B,)
    return scores.detach().cpu().numpy().astype(np.float32)


payload = task.evaluate(model_fn)
run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)
print(f"Benchmark written to {run_dir / 'benchmark.json'}")
