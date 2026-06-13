import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"


@dataclass
class AttentionSpanModel:
    """Minimal attention model: token + pos embeddings -> single attention head.

    The needle token (9999) gets a strong, distinctive embedding so the query
    position attends to it; we return the full attention matrix so the task can
    read attention from the query (pos 0) to the needle at distance d.
    """
    vocab_size: int = 10000  # needle token 9999, query 8888, distractors 1..999
    d_model: int = 64
    seq_len: int = 512
    key_token: int = 9999

    def __post_init__(self):
        self.token_emb = nn.Embedding(self.vocab_size, self.d_model).to(DEVICE)
        self.pos_emb = nn.Embedding(self.seq_len, self.d_model).to(DEVICE)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False).to(DEVICE)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False).to(DEVICE)

        # Initialize so key token has a strong, distinctive key vector
        with torch.no_grad():
            # Make key token embedding stand out
            self.token_emb.weight[self.key_token].normal_(0, 10.0)
            # Q/K projections: identity-ish so token identity drives attention
            self.q_proj.weight.copy_(torch.eye(self.d_model, device=DEVICE))
            self.k_proj.weight.copy_(torch.eye(self.d_model, device=DEVICE))

    def parameters(self):
        for m in (self.token_emb, self.pos_emb, self.q_proj, self.k_proj):
            yield from m.parameters()

    def eval(self):
        for m in (self.token_emb, self.pos_emb, self.q_proj, self.k_proj):
            m.eval()
        return self

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns full attention weights [batch, seq_len, seq_len]."""
        batch, seq_len = input_ids.shape

        # Embeddings
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)  # [batch, seq, d_model]

        # Project to Q, K
        q = self.q_proj(x)  # [batch, seq, d_model]
        k = self.k_proj(x)  # [batch, seq, d_model]

        # Full attention matrix: each query position attends over all keys.
        all_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5)  # [batch, seq, seq]
        attn = torch.softmax(all_scores, dim=-1)  # [batch, seq, seq]
        return attn  # [batch, seq, seq]


def make_model_fn(model: AttentionSpanModel):
    """Wrap the torch model into the numpy callable expected by task.evaluate."""
    model.eval()

    def model_fn(input_ids: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(input_ids, device=DEVICE)
            attn = model.forward(tensor)  # [batch, seq, seq]
            return attn.detach().cpu().numpy().astype(np.float32)

    return model_fn


def main():
    task = load_task(__file__)

    # Build and (optionally) load a checkpoint — here we just use the initialized model
    model = AttentionSpanModel()
    model_fn = make_model_fn(model)

    # Override model_name in payload after evaluation
    payload = task.evaluate(model_fn)
    payload["model_name"] = "first_pass_initialized"

    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":
    main()