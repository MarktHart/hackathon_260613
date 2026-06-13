import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from agentic.experiments import load_task, record_benchmark, results_dir

# ----------------------------------------------------------------------
# Constants from task.py (must match exactly)
# ----------------------------------------------------------------------
VOCAB_SIZE = 15
SEQ_LEN = 14
MAX_DIGITS = 3
SUM_DIGITS = 4
SUM_START_IDX = 9  # 2 * MAX_DIGITS + 3
SUM_POSITIONS = list(range(SUM_START_IDX, SUM_START_IDX + SUM_DIGITS))

# Special token IDs
PLUS_TOKEN = 10
EQUALS_TOKEN = 11
BOS_TOKEN = 12
EOS_TOKEN = 13
PAD_TOKEN = 14

# ----------------------------------------------------------------------
# Minimal Transformer (base_model.py style)
# ----------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (batch, seq, d_model)
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.mlp = MLP(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class AdditionTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        seq_len: int = SEQ_LEN,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (optional but common)
        self.head.weight = self.token_emb.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, T)
        B, T = input_ids.shape
        device = input_ids.device
        
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)
        
        # Causal mask for autoregressive generation (not strictly needed here but standard)
        mask = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)
        
        for block in self.blocks:
            x = block(x, mask)
            
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
DEVICE = "cuda"
BATCH_SIZE = 64
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 0.01

def train_model(model: AdditionTransformer, task, seed: int = 0) -> AdditionTransformer:
    """Train the model on addition problems."""
    model.train()
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Generate training data
    batch = task.generate(seed=seed)
    input_ids = torch.from_numpy(batch.input_ids).long().to(DEVICE)
    target_digits = torch.from_numpy(batch.target_sum_digits).long().to(DEVICE)
    
    # We only compute loss on SUM positions
    sum_positions = torch.tensor(SUM_POSITIONS, device=DEVICE)
    
    n_samples = input_ids.shape[0]
    indices = torch.arange(n_samples)
    
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, n_samples, BATCH_SIZE):
            batch_idx = perm[i:i+BATCH_SIZE]
            x = input_ids[batch_idx]
            y = target_digits[batch_idx]  # (B, SUM_DIGITS)
            
            logits = model(x)  # (B, SEQ_LEN, VOCAB_SIZE)
            sum_logits = logits[:, SUM_POSITIONS, :]  # (B, SUM_DIGITS, VOCAB_SIZE)
            
            loss = F.cross_entropy(
                sum_logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / max(1, num_batches)
        
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch}: loss = {avg_loss:.4f}")
    
    return model


# ----------------------------------------------------------------------
# model_fn for evaluation (must run on GPU)
# ----------------------------------------------------------------------
def make_model_fn(model: AdditionTransformer):
    """Wrap the trained model into the model_fn signature expected by task.evaluate."""
    model.eval()
    model.to(DEVICE)
    
    def model_fn(input_ids: np.ndarray) -> np.ndarray:
        # input_ids: (N, SEQ_LEN) numpy array
        with torch.no_grad():
            x = torch.from_numpy(input_ids).long().to(DEVICE)
            logits = model(x)  # (N, SEQ_LEN, VOCAB_SIZE)
            return logits.detach().cpu().numpy().astype(np.float64)
    
    return model_fn


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Load the task
    task = load_task(__file__)
    
    # Create and train model
    print("Creating model...")
    model = AdditionTransformer()
    
    print("Training...")
    model = train_model(model, task, seed=0)
    
    # Save checkpoint
    run_dir = results_dir(__file__)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Evaluate
    print("Evaluating...")
    model_fn = make_model_fn(model)
    payload = task.evaluate(model_fn)
    
    # Record benchmark
    record_benchmark(__file__, run_dir, payload)
    print("Done!")
    print(f"Results in {run_dir}")