# main.py for 'pass_3' attempt at the attention_span goal.
# Goal: measure how far back attention can decay from a fixed query at position 0 to a needle at distance d.

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

# Architecture: a minimal single-head attention block with an explicit learnable
# denominator (softness) on the attention logits, plus an MLP. Runs on CUDA.

DEVICE = torch.device("cuda")
print(f"Using device: {DEVICE}")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len_max: int = 512):
        super().__init__()
        pe = torch.zeros(seq_len_max, d_model)
        position = torch.arange(0, seq_len_max, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer("pe", pe)  # fixed during training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.shape[1], :]


class AttentionSpanModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64):
        super().__init__()
        self.vocab_size, self.d_model = vocab_size, d_model
        self.seq_len_max = 512
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, seq_len_max=self.seq_len_max)

        # Single self-attention head.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable denominator (softness) scalar applied to attention logits.
        self.log_denom = nn.Parameter(torch.zeros(1))

        # Simple MLP after attention.
        hidden_dim = d_model * 4
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model, bias=False),
        )
        self.post_attn_scale = nn.Parameter(0.1 * torch.ones(1))
        self.post_mlp_scale = nn.Parameter(0.1 * torch.ones(1))

    def _attn(self, input_ids: torch.Tensor):
        batch, seq_len = input_ids.shape
        if seq_len > self.seq_len_max:
            raise ValueError(f"Sequence length {seq_len} exceeds seq_len_max={self.seq_len_max}")

        x = self.embedding(input_ids)          # (b, L, d)
        x = self.positional_encoding(x)        # (b, L, d)

        q = self.q_proj(x)                     # (b, L, d)
        k = self.k_proj(x)                     # (b, L, d)
        v = self.v_proj(x)                     # (b, L, d)

        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)  # (b, L, L)
        attn_score = attn_score - self.log_denom  # explicit learnable denominator
        attn = F.softmax(attn_score, dim=-1)   # (b, L, L)

        out = self.out_proj(torch.matmul(attn, v))  # (b, L, d)
        x = x + self.post_attn_scale * out
        residual_mlp = x
        x = residual_mlp + self.post_mlp_scale * self.mlp(x)
        return attn, x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Return attention weights of shape (batch, seq_len, seq_len)
        attn, _ = self._attn(input_ids)
        return attn


def train_model(vocab_size: int, d_model: int = 64, num_steps: int = 50):
    seq_len = 512
    canonical_distances = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    batch_size = 32

    model = AttentionSpanModel(vocab_size, d_model=d_model).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(num_steps):
        # Build a small training batch: query at 0, needle at a random canonical distance.
        seqs = torch.zeros(batch_size, seq_len, dtype=torch.long, device=DEVICE)
        targets = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)
        for b in range(batch_size):
            d = canonical_distances[np.random.randint(len(canonical_distances))]
            seqs[b, 0] = 8888   # query token
            seqs[b, d] = 9999   # needle token
            targets[b] = d

        optimizer.zero_grad()
        attn = model(seqs)                       # (b, L, L)
        # Encourage attention from query (pos 0) onto the needle position.
        attn_q = attn[:, 0, :]                    # (b, L)
        loss = F.cross_entropy(torch.log(attn_q + 1e-9), targets)
        loss.backward()
        optimizer.step()
        if (step + 1) % 25 == 0:
            print(f"step {step+1}: loss = {loss.item():.4e}")

    return model


def evaluate_and_save(model: AttentionSpanModel) -> dict:
    def model_fn(input_ids: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            ids = torch.as_tensor(input_ids, dtype=torch.long, device=DEVICE)
            # Process in chunks to bound memory (full attn is (B, 512, 512)).
            outs = []
            chunk = 64
            for start in range(0, ids.shape[0], chunk):
                attn = model(ids[start:start + chunk])  # (b, L, L)
                outs.append(attn.detach().cpu().numpy().astype(np.float32))
            return np.concatenate(outs, axis=0)

    payload = task.evaluate(model_fn)
    payload["model_name"] = "attention_span_model"
    return payload


def main():
    print("Starting training...")
    model = train_model(vocab_size=10000, d_model=64, num_steps=50)
    print("Evaluating...")
    result = evaluate_and_save(model)

    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, result)
    print(f"Results written to {run_dir}")


if __name__ == "__main__":
    main()
