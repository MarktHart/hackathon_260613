import torch
import torch.nn as nn

# Base import of the synthetic task generator + evaluator (pure NumPy).
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

task = load_task(__file__)


# ----------------------------------------------------------------------
# 1. Model design
#    Small delta from experiments/base_model.py to solve the goal.
# ----------------------------------------------------------------------
class AttentionArgminHead(nn.Module):
    """
    A single attention head wired to approximate argmin over per-position
    values. Q, K, V all derived from the scalar `values` vector.
    """
    def __init__(self, seq_len=64, key_dim=32, proj_dim=4):
        super().__init__()
        # Tiny projection to give us a small trainable buffer before the
        # attention core: map each scalar value to a key vector.
        self.value_proj = nn.Linear(1, key_dim)

        # Standard query vector — set to (1, 0, …, 0) initially.
        self.query = nn.Parameter(torch.eye(key_dim)[0])

    def forward(self, keys, values, query):
        # keys: (seq_len, key_dim)  — synthetic keys that are unit vectors
        # values: (seq_len,)        — scalar floats
        # query: (key_dim,)         — fixed first-row basis vector

        # Project scalar value at each position through a tiny linear layer
        # to get a K vector we can use for the attention softmax.
        K = self.value_proj(values[:, None])   # (L, key_dim)

        # Compute logits per position: query (key_dim,) dot K (L, key_dim) -> (L,)
        attn_logits = torch.einsum('d,ld->l', query, K)   # (L,)

        # Softmax over positions yields the attention distribution.
        attn = torch.nn.functional.softmax(attn_logits, dim=-1)   # (L,)
        return attn   # (L,) attention weights


# ----------------------------------------------------------------------
# 2. Evaluate the model on the synthetic batch
# ----------------------------------------------------------------------
def model_fn(keys, values, query):
    # Cast to float32 on the GPU as expected.
    keys = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)
    values = torch.as_tensor(values, dtype=torch.float32, device=DEVICE)
    query = torch.as_tensor(query, dtype=torch.float32, device=DEVICE)

    # Small projection dimension gives us a tiny trainable buffer.
    head = AttentionArgminHead(key_dim=query.shape[0]).to(DEVICE)

    # The forward pass returns attention weights of shape (seq_len,).
    out = head(keys, values, query)
    return out.detach().cpu().numpy()


# ----------------------------------------------------------------------
# 3. Driver that runs the canonical sweep
# ----------------------------------------------------------------------
payload = task.evaluate(model_fn)

# Write results under results/<timestamp>/...
run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)