import math

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = torch.device("cuda")
print(f"Using device: {DEVICE}")

# Load the task to get the canonical sweep and metric.
task = load_task(__file__)


class AttentionSpanModel(torch.nn.Module):
    """Minimal transformer head: Q / K / V projection, optional positional
    encoding, scaled softmax attention, and an explicitlearnable scale on
    the attention logits (to control softness). No MLP after attention.
    """
    def __init__(self, vocab_size: int, d_model: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.k_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.log_denom = torch.nn.Parameter(torch.zeros(1))  # Learnable softness denominator

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len)
        # Returns attention weights of shape (batch, num_heads, seq_len, seq_len).
        # We return a single head, so the output is (batch, seq_len, seq_len).

        batch, seq_len = input_ids.shape
        x = self.embedding(input_ids)               # (B, L, D)

        q = self.q_proj(x)                           # (B, L, D)
        k = self.k_proj(x). transpose(-2, -1)        # (B, D, L)

        # Attention logits
        scores = (q @ k) / math.sqrt(self.d_model)   # (B, L, L)

        # Apply learnable softening denominator
        scores = scores - self.log_denom

        attn = torch.softmax(scores, dim=-1)         # (B, L, L)

        return attn.unsqueeze(1)                     # (B, 1, L, L) → 4D as required


def evaluate_model(vocab_size: int, num_steps: int = 50, lr: float = 1e-2):
    """Train a single attention head to focus from position 0 on a needle
    that moves to canonical distances: 1, 2, 4, ..., 256.
    """
    model = AttentionSpanModel(vocab_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    canonical_distances = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    batch_size = 32

    for step in range(num_steps):
        # Build batch: query token 8888 at position 0, needle token 9999 at a random canonical distance d.
        seqs = torch.zeros(batch_size, 512, dtype=torch.long, device=DEVICE)
        targets = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)
        for b in range(batch_size):
            d = canonical_distances[np.random.randint(len(canonical_distances))]
            seqs[b, 0] = 8888   # query
            seqs[b, d] = 9999   # needle
            targets[b] = d

        optimizer.zero_grad()
        attn = model(seqs)               # (B, 1, 512, 512)
        attn_q = attn[:, 0, 0, :]        # (B, 512) attention from query position 0 to all keys
        # Cross-entropy pushes the weight onto the needle position.
        loss = torch.nn.functional.cross_entropy(attn_q, targets)
        loss.backward()
        optimizer.step()

        if (step + 1) % 25 == 0:
            print(f"step {step+1}: loss = {loss.item():.4e}")

    return model


def main():
    print("Training minimal attention head...")
    model = evaluate_model(vocab_size=10000, num_steps=50)

    # Evaluation: run through the canonical batch and return the model function.
    def model_fn(input_ids: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            ids = torch.as_tensor(input_ids, dtype=torch.long, device=DEVICE)
            # Process in chunks to avoid OOM on a 4D attention tensor of shape (900, 1, 512, 512).
            chunks = []
            chunk = 64
            for start in range(0, ids.shape[0], chunk):
                end = min(start + chunk, ids.shape[0])
                out = model(ids[start:end])
                chunks.append(out.cpu().numpy())
            result = np.concatenate(chunks, axis=0)   # (900, 1, 512, 512)
            return result

    print("Computing payload...")
    payload = task.evaluate(model_fn)
    payload["model_name"] = "single_head_attention_span_model"

    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    print(f"Results written to {run_dir}")


if __name__ == "__main__":
    main()