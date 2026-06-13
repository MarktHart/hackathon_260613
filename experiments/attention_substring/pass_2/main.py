import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from agentic.experiments import load_task, record_benchmark, results_dir

device = torch.device("cuda")

VOCAB_SIZE = 64
D_MODEL = 32
D_HEAD = 16
N_LAYERS = 2
N_HEADS = 1


def _build_pos_emb(seq_len: int, d_model: int) -> torch.Tensor:
    pos_emb = torch.zeros((seq_len, d_model), device=device)
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.pow(
        10000.0,
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device) / d_model,
    )
    pos_emb[:, 0::2] = torch.sin(position / div)
    pos_emb[:, 1::2] = torch.cos(position / div)
    return pos_emb


class SubstringModel(nn.Module):
    """A tiny 2-layer single-head attention model on CUDA."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.q = nn.ModuleList([nn.Linear(D_MODEL, D_HEAD, bias=False) for _ in range(N_LAYERS)])
        self.k = nn.ModuleList([nn.Linear(D_MODEL, D_HEAD, bias=False) for _ in range(N_LAYERS)])
        self.v = nn.ModuleList([nn.Linear(D_MODEL, D_HEAD, bias=False) for _ in range(N_LAYERS)])
        self.o = nn.ModuleList([nn.Linear(D_HEAD, D_MODEL, bias=False) for _ in range(N_LAYERS)])
        self.mlp1 = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL * 4) for _ in range(N_LAYERS)])
        self.mlp2 = nn.ModuleList([nn.Linear(D_MODEL * 4, D_MODEL) for _ in range(N_LAYERS)])
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, input_ids):
        # input_ids: [1, seq_len]
        seq_len = input_ids.shape[1]
        x = self.embed(input_ids) + _build_pos_emb(seq_len, D_MODEL)  # [1, L, d_model]

        attn_per_layer = []
        for li in range(N_LAYERS):
            q = self.q[li](x)  # [1, L, d_head]
            k = self.k[li](x)
            v = self.v[li](x)
            scores = torch.matmul(q, k.transpose(-1, -2)) / (D_HEAD ** 0.5)  # [1, L, L]
            attn = F.softmax(scores, dim=-1)  # [1, L, L]
            attn_per_layer.append(attn)
            out = self.o[li](torch.matmul(attn, v))  # [1, L, d_model]
            x = x + out
            x = x + self.mlp2[li](F.relu(self.mlp1[li](x)))  # residual MLP

        logits = self.lm_head(x).squeeze(0)  # [L, vocab]
        attn = torch.stack(attn_per_layer, dim=0)  # [n_layers, 1, L, L]
        return attn, logits


_model = SubstringModel().to(device)
_model.eval()


def model_fn(input_ids: np.ndarray) -> dict:
    with torch.no_grad():
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=device)  # [1, L]
        attn, logits = _model(ids)
        return {
            "attn_weights": attn.detach().cpu().numpy().astype(np.float32),  # [n_layers, n_heads, L, L]
            "logits": logits.detach().cpu().numpy().astype(np.float32),      # [L, vocab]
        }


def main():
    task = load_task(__file__)
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    print(f"Benchmark written to {run_dir / 'benchmark.json'}")


if __name__ == "__main__":
    main()
