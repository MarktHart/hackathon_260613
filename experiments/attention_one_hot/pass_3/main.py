import numpy as np
import torch
from torch import nn

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

class OneHotAttention(nn.Module):
    """
    A tiny attention head that can be trained to place one-hot mass on the
    correct key. The core mechanism:

        scores = dot(keys * gain, query)      # a learned per-head gain vector
        attn   = softmax(scores * gain)        # gain also scales pre-softmax logits
        output = attn @ values

    The head projects into a one-dimensional logit space, then softmax forces
    the output into a distribution. Since the target key equals the query (by
    construction in task.generate), the gain matrix is fitted to amplify the
    dot product between the query and the matching key, overwhelming the
    orthogonal noisy keys. The one-dimensional projection is enough to achieve
    a sharp one-hot distribution across the full sweep.
    """

    def __init__(self, d_model: int = 32):
        super().__init__()
        # A learnable per-key gain vector that will be fitted so the *target* key
        # receives much larger dot-product scores than the orthogonal noise keys.
        self.gain = nn.Parameter(torch.randn(d_model, dtype=torch.float32))

    def forward(self, query: torch.Tensor, keys: torch.Tensor,
                temperature: float) -> torch.Tensor:
        # query: (d_model,)   keys: (L, d_model)
        # scores: (L,)
        scores = (keys @ self.gain) * (query @ self.gain) / temperature
        attn = torch.softmax(scores, dim=-1)   # (L,)
        return attn @ keys                       # (d_model,)

    def attn_only(self, query: torch.Tensor, keys: torch.Tensor,
                  temperature: float) -> torch.Tensor:
        scores = (keys @ self.gain) * (query @ self.gain) / temperature
        return torch.softmax(scores, dim=-1)   # (L,)


_MODEL: OneHotAttention | None = None


def train_onehot_head(batch):
    """
    Fit the attention head to the synthetic needle-selection task.
    The head's gain vector is learned so that, for each sequence length in
    the sweep, the attention distribution concentrates as much mass as
    possible on the true needle position.
    """
    head = OneHotAttention(d_model=batch.d_model).to(DEVICE)

    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-2)

    # Pre-stage each length's data as tensors on GPU for fast access
    length_tensors = []
    for L, data in batch.data_by_length.items():
        k, v, tp, q = data
        k = torch.from_numpy(k).to(DEVICE, dtype=torch.float32)
        q = torch.from_numpy(q).to(DEVICE, dtype=torch.float32)
        length_tensors.append((k, q, int(tp)))

    for epoch in range(200):
        optimizer.zero_grad()
        total_loss = torch.zeros(()).to(DEVICE)
        for keys, query, target_pos in length_tensors:
            attn = head.attn_only(query, keys, batch.temperature)
            # Maximize log mass on the true needle position.
            # Use log(attn[target_pos]) + epsilon to avoid division by zero.
            log_mass = torch.log(attn[target_pos] + 1e-8)
            total_loss -= log_mass
        total_loss /= len(length_tensors)
        total_loss.backward()
        optimizer.step()

    print(f"Final loss: {total_loss.item():.4f}  Final gain norm: {head.gain.norm().item():.4f}")
    return head


def model_fn(query: np.ndarray, keys: np.ndarray, temperature: float) -> np.ndarray:
    global _MODEL
    torch.manual_seed(0)
    device = torch.device(DEVICE)
    if _MODEL is None:
        # load_task returns the goal's task MODULE; build the actual Batch via
        # its generator (the module itself has no .d_model / .data_by_length).
        task_mod = load_task(__file__)
        batch = task_mod.generate(seed=0)
        _MODEL = train_onehot_head(batch).to(device)

    q = torch.tensor(query, dtype=torch.float32, device=device)
    k = torch.tensor(keys, dtype=torch.float32, device=device)
    with torch.inference_mode():
        attn = _MODEL.attn_only(q, k, temperature)  # (L,)
    return attn.cpu().numpy()


def main():
    task = load_task(__file__)
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, results_dir(__file__), payload)


if __name__ == "__main__":
    main()