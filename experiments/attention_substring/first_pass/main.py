import numpy as np
import torch
from torch import nn

from agentic.experiments import load_task, record_benchmark, results_dir

device = torch.device("cuda")

VOCAB_SIZE = 64
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 1


class SimpleAttentionHead(nn.Module):
    def __init__(self, d_model=64, d_hidden=16):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.qkv_proj = nn.Linear(d_model, 3 * d_hidden, bias=False)
        self.out_proj = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        qkv = self.qkv_proj(x)                      # [b, L, 3*d_hidden]
        q, k, v = qkv.chunk(3, dim=-1)              # each [b, L, d_hidden]
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_hidden ** 0.5)  # [b, L, L]
        attn_weights = attn_scores.softmax(dim=-1)  # [b, L, L]
        attn_out = torch.matmul(attn_weights, v)    # [b, L, d_hidden]
        out = self.out_proj(attn_out)               # [b, L, d_model]
        return out, attn_weights


class SubstringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.heads = nn.ModuleList([SimpleAttentionHead(D_MODEL) for _ in range(N_LAYERS)])
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, input_ids):
        # input_ids: [1, seq_len]
        x = self.embed(input_ids)  # [1, L, d_model]
        attn_per_layer = []
        for head in self.heads:
            out, attn_weights = head(x)
            x = x + out  # residual
            attn_per_layer.append(attn_weights)  # [1, L, L]
        logits = self.lm_head(x)  # [1, L, vocab]
        # Stack attn to [n_layers, n_heads, L, L]
        attn = torch.stack(attn_per_layer, dim=0)  # [n_layers, 1, L, L]
        return attn, logits.squeeze(0)  # [n_layers,1,L,L], [L, vocab]


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
