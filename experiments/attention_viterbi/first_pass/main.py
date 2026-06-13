import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from agentic.experiments import load_task, record_benchmark, results_dir

# Load the goal's task module (data generator + evaluator)
task = load_task(__file__)

# ------------------------------------------------------------
# Config (pulled from task so it stays in sync)
# ------------------------------------------------------------
SEQ_LEN      = task.SEQ_LEN          # 20
N_OBS        = task.N_OBS            # 4
N_EVAL_SEQ   = task.N_EVAL_SEQ       # 100
EVAL_SEED    = task.EVAL_SEED        # 42
MODEL_CONFIG = task.MODEL_CONFIG
N_LAYERS     = MODEL_CONFIG["n_layers"]     # 2
N_HEADS      = MODEL_CONFIG["n_heads"]      # 4
D_MODEL      = MODEL_CONFIG["d_model"]      # 64
VOCAB_SIZE   = MODEL_CONFIG["vocab_size"]   # 4

# HMM parameters (for training data generation)
HMM_PI = task.HMM_PI
HMM_A  = task.HMM_A
HMM_B  = task.HMM_B
N_STATES = task.N_STATES

DEVICE = "cuda"          # GPU is guaranteed by the pipeline
BATCH_SIZE = 64
LR = 3e-4
N_EPOCHS = 30            # enough for a clean signal on this tiny task

# ------------------------------------------------------------
# Model: attention-only causal transformer (2L, 4H, d=64)
# ------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)).view(1, 1, SEQ_LEN, SEQ_LEN))

    def forward(self, x):                      # x: [B, T, D]
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, self.n_heads, 3 * self.d_head).transpose(1, 2)  # [B, H, T, 3Dh]
        q, k, v = qkv.chunk(3, dim=-1)         # each [B, H, T, Dh]
        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)  # [B, H, T, T]
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, D]
        out = self.out(out)
        return out, att                        # return weights for the payload


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.ln(x)
        x, attn = self.attn(x)
        x = x + residual
        return x, attn


class AttentionOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):                    # idx: [B, T] long
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)   # [1, T]
        x = self.tok_emb(idx) + self.pos_emb(pos)               # [B, T, D]
        all_attn = []
        for blk in self.blocks:
            x, attn = blk(x)
            all_attn.append(attn)
        x = self.ln_f(x)
        logits = self.head(x)                   # [B, T, V]
        # stack attn: [B, n_layers, n_heads, T, T]
        attn_stack = torch.stack(all_attn, dim=1)
        return logits, attn_stack


# ------------------------------------------------------------
# Training data generator (same HMM as task, more sequences)
# ------------------------------------------------------------
def generate_training_data(n_sequences, seq_len, seed=0):
    rng = np.random.default_rng(seed)
    data = np.zeros((n_sequences, seq_len), dtype=np.int64)
    for i in range(n_sequences):
        states = np.zeros(seq_len, dtype=np.int64)
        states[0] = rng.choice(N_STATES, p=HMM_PI)
        for t in range(1, seq_len):
            states[t] = rng.choice(N_STATES, p=HMM_A[states[t-1]])
        for t in range(seq_len):
            data[i, t] = rng.choice(N_OBS, p=HMM_B[states[t]])
    return data


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------
def train_model(model, train_data, epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    model.to(DEVICE)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_samples = train_data.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            batch = torch.from_numpy(train_data[idx]).to(DEVICE)   # [B, T]
            logits, _ = model(batch[:, :-1])                       # predict next token
            targets = batch[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}  avg loss: {total_loss/n_batches:.4f}")


# ------------------------------------------------------------
# model_fn for task.evaluate  (must run on GPU, return NumPy)
# ------------------------------------------------------------
def make_model_fn(model):
    model.to(DEVICE)
    model.eval()

    @torch.no_grad()
    def model_fn(input_ids: np.ndarray) -> dict[str, np.ndarray]:
        """
        input_ids: [batch, seq_len] int32/64, values 0..3
        Returns: {
            "attn_weights": [batch, n_layers, n_heads, seq_len, seq_len] float32,
            "logits":       [batch, seq_len, vocab_size] float32
        }
        """
        x = torch.as_tensor(input_ids, dtype=torch.long, device=DEVICE)  # [B, T]
        logits, attn = model(x)                      # attn: [B, L, H, T, T]
        return {
            "attn_weights": attn.detach().cpu().numpy().astype(np.float32),
            "logits":       logits.detach().cpu().numpy().astype(np.float32),
        }
    return model_fn


# ------------------------------------------------------------
# Main entry
# ------------------------------------------------------------
def main():
    print("Loading task...")
    # Canonical eval batch (seed=42) is produced inside task.evaluate

    print("Generating training data...")
    train_data = generate_training_data(n_sequences=5000, seq_len=SEQ_LEN, seed=123)

    print("Initializing model...")
    model = AttentionOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        seq_len=SEQ_LEN,
    )

    print("Training...")
    train_model(model, train_data)

    print("Evaluating...")
    model_fn = make_model_fn(model)
    payload = task.evaluate(model_fn)

    run_dir = results_dir(__file__)
    run_dir.mkdir(parents=True, exist_ok=True)
    record_benchmark(__file__, run_dir, payload)
    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":
    main()