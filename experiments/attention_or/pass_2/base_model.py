import torch
import torch.nn as nn

class BaseTransformer(nn.Module):
    """Minimal single-layer transformer: one self-attention block plus one MLP block."""
    def __init__(self, d: int = 32, n: int = 64):
        super().__init__()
        # Attention block (Q, K, V projections + softmax + out projection)
        self.attn = nn.Sequential(
            nn.Linear(d, d, bias=False),   # query projection
            nn.Linear(d, d, bias=False),   # key projection
            nn.Linear(d, d, bias=False),   # value projection
            nn.Softmax(dim=-1),             # scaled_softmax is applied upstream in the toy head
            nn.Linear(d, d, bias=False),   # out projection
        )
        # MLP block (FFN with GeLU activation)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d, bias=False),
            nn.GELU(),
            nn.Linear(4 * d, d, bias=False),
        )

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        # query shape (d,), keys shape (d, n)
        q = query[None]  # (1, d)
        k = keys.permute(1, 0)  # (n, d)
        v = keys.permute(1, 0)  # (n, d)
        # Self-attention: compute attention scores and apply output projection
        attn_scores = (q @ keys) / keys.shape[0]  # (1, n)
        attn_out = attn_scores @ v  # (1, d)
        attn_out = self.attn[-1](attn_out)  # final out projection
        # MLP block
        mlp_out = self.mlp(attn_out)
        return mlp_out[0]  # (d,))