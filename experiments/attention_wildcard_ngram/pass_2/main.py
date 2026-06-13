import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir
from torch import nn

# Load task metadata (vocab size, seq len from the canonical batch)
task = load_task(__file__)

DEVICE = "cuda"


class WildcardNgramModel(nn.Module):
    """A tiny single-head attention model that, given the sequence, produces an
    attention matrix (n_sequences, seq_len, seq_len). The anchor token gets a
    distinctive embedding so the target query attends to it, skipping the
    wildcard span.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 16, anchor_token: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.anchor_token = anchor_token
        self.tokenizer = nn.Embedding(vocab_size, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        with torch.no_grad():
            # Identity-ish projections so token identity drives attention.
            self.q_proj.weight.copy_(torch.eye(embed_dim))
            self.k_proj.weight.copy_(torch.eye(embed_dim))
            # Make the anchor token's embedding stand out strongly.
            self.tokenizer.weight[anchor_token].normal_(0.0, 10.0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [N, L]
        x = self.tokenizer(input_ids)            # [N, L, d]
        q = self.q_proj(x)                       # [N, L, d]
        k = self.k_proj(x)                       # [N, L, d]
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.embed_dim ** 0.5)  # [N, L, L]
        attn = torch.softmax(scores, dim=-1)     # [N, L, L]
        return attn


_anchor_token = getattr(task, "ANCHOR_TOKEN", 1)
_vocab_size = getattr(task, "VOCAB_SIZE", 32)
_model = WildcardNgramModel(_vocab_size, anchor_token=_anchor_token).to(DEVICE)
_model.eval()


def model_fn(batch) -> np.ndarray:
    """task contract: model_fn(batch) -> attention (n_sequences, seq_len, seq_len)."""
    with torch.inference_mode():
        ids = torch.as_tensor(batch.sequences, dtype=torch.long, device=DEVICE)
        attn = _model(ids)  # [N, L, L]
        return attn.detach().cpu().numpy().astype(np.float32)


def run():
    run_dir = results_dir(__file__)
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, run_dir, payload)
    print(f"Benchmark written to {run_dir / 'benchmark.json'}")


if __name__ == "__main__":
    run()
