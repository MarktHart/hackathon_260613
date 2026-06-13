import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from agentic.experiments import load_task, record_benchmark, results_dir

# ---- Config -----------------------------------------------------------------
DEVICE = "cuda"
assert torch.cuda.is_available(), "GPU required for this attempt"

SEQ_LEN = 32
NUM_SEQUENCES = 500
VOCAB_SIZE = 104
N_FILLER = 100
OPEN_A, CLOSE_A = 100, 101
OPEN_B, CLOSE_B = 102, 103
BRACKET_TOKENS = {OPEN_A, CLOSE_A, OPEN_B, CLOSE_B}

# Model hyperparameters
N_LAYERS = 2
N_HEADS = 8
D_MODEL = 128
D_HEAD = D_MODEL // N_HEADS
DROPOUT = 0.1

# Training config
TRAIN_STEPS = 2000
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 0.01

# ---- Model: small transformer with attention weight output -------------------
class BracketTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        
        self.layers = nn.ModuleList([
            TransformerBlock(D_MODEL, N_HEADS, DROPOUT) for _ in range(N_LAYERS)
        ])
        self.ln_f = nn.LayerNorm(D_MODEL)
        
        # Store attention weights for inspection
        self.attn_weights = []  # will hold list of [B, H, S, S] per layer
    
    def forward(self, input_ids: torch.Tensor, return_attn: bool = False):
        B, S = input_ids.shape
        self.attn_weights = []
        
        x = self.token_emb(input_ids) + self.pos_emb(torch.arange(S, device=DEVICE))
        
        for layer in self.layers:
            x, attn = layer(x, return_attn=True)
            if return_attn:
                self.attn_weights.append(attn)  # [B, H, S, S]
        
        x = self.ln_f(x)
        
        if return_attn:
            # Stack: [B, L, H, S, S]
            return x, torch.stack(self.attn_weights, dim=1)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, return_attn: bool = False):
        residual = x
        x = self.ln1(x)
        x, attn = self.attn(x, return_attn=True)
        x = residual + self.dropout(x)
        
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        
        if return_attn:
            return x, attn
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, S, _ = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.n_heads, self.d_head).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(0)  # each [B, S, H, D]
        
        # [B, H, S, S]
        attn_scores = torch.einsum('bshd,bthd->bhst', q, k) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.einsum('bhst,bthd->bshd', attn_weights, v)
        out = out.contiguous().view(B, S, -1)
        out = self.out(out)
        
        if return_attn:
            return out, attn_weights
        return out, None


# ---- Constraint-aware loss ---------------------------------------------------
def constraint_loss(attn_weights: torch.Tensor, batch_constraints: list, seq_len: int) -> torch.Tensor:
    """
    attn_weights: [B, L, H, S, S]
    batch_constraints: list of list of (i, j, d) directed entries per sequence
    """
    B, L, H, S, _ = attn_weights.shape
    total_loss = 0.0
    total_pairs = 0
    
    for b in range(B):
        constraints = batch_constraints[b]
        if not constraints:
            continue
        for (i, j, d) in constraints:
            # Want high attention from i to j across all layers/heads
            # Use max over heads, mean over layers
            attn_ij = attn_weights[b, :, :, i, j]  # [L, H]
            # Encourage at least one head in at least one layer to attend strongly
            max_attn = attn_ij.max()
            # Negative log likelihood style: want max_attn -> 1
            total_loss += -torch.log(max_attn + 1e-8)
            total_pairs += 1
    
    if total_pairs == 0:
        return torch.tensor(0.0, device=attn_weights.device)
    return total_loss / total_pairs


# ---- Data generation (mirrors task.py) --------------------------------------
def generate_batch(batch_size: int, seq_len: int = SEQ_LEN, seed: int = None) -> tuple:
    """Generate a batch of sequences with bracket constraints."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    DISTANCES = (1, 2, 4, 8, 12, 16)
    CONSTRAINT_TYPES = 2
    BRACKETS = {0: (OPEN_A, CLOSE_A), 1: (OPEN_B, CLOSE_B)}
    
    all_tokens = []
    all_constraints = []
    
    for _ in range(batch_size):
        occupied = np.zeros(seq_len, dtype=bool)
        tokens = rng.integers(0, N_FILLER, size=seq_len).astype(np.int32)
        directed = []
        
        for ctype in range(CONSTRAINT_TYPES):
            open_tok, close_tok = BRACKETS[ctype]
            n_pairs = int(rng.integers(2, 5))
            for _ in range(n_pairs):
                placed = False
                for _try in range(40):
                    d = int(rng.choice(DISTANCES))
                    if d >= seq_len:
                        continue
                    o = int(rng.integers(0, seq_len - d))
                    c = o + d
                    if occupied[o] or occupied[c]:
                        continue
                    occupied[o] = occupied[c] = True
                    tokens[o] = open_tok
                    tokens[c] = close_tok
                    directed.append((o, c, d))
                    directed.append((c, o, d))
                    placed = True
                    break
                if not placed:
                    continue
        
        all_tokens.append(tokens)
        all_constraints.append(directed)
    
    input_ids = torch.from_numpy(np.stack(all_tokens)).to(DEVICE)
    return input_ids, all_constraints


# ---- Training ---------------------------------------------------------------
def train_model(model: BracketTransformer, steps: int = TRAIN_STEPS):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    
    print(f"Training for {steps} steps on {DEVICE}...")
    for step in range(steps):
        opt.zero_grad()
        input_ids, constraints = generate_batch(BATCH_SIZE)
        _, attn_weights = model(input_ids, return_attn=True)  # [B, L, H, S, S]
        loss = constraint_loss(attn_weights, constraints, SEQ_LEN)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        
        if step % 200 == 0:
            print(f"  Step {step:4d} | loss: {loss.item():.4f}")
    
    print("Training complete.")
    return model


# ---- model_fn for evaluation ------------------------------------------------
def make_model_fn(model: BracketTransformer):
    model.eval()
    
    def model_fn(input_ids: np.ndarray) -> np.ndarray:
        """Returns attention weights [batch, n_layers, n_heads, seq_len, seq_len]"""
        with torch.no_grad():
            ids = torch.as_tensor(input_ids, dtype=torch.int64, device=DEVICE)
            _, attn = model(ids, return_attn=True)  # [B, L, H, S, S]
            return attn.detach().cpu().numpy().astype(np.float32)
    
    return model_fn


# ---- Main -------------------------------------------------------------------
if __name__ == "__main__":
    # Load task
    task = load_task(__file__)
    
    # Build and train model
    model = BracketTransformer().to(DEVICE)
    model = train_model(model, TRAIN_STEPS)
    
    # Create model_fn and evaluate
    model_fn = make_model_fn(model)
    
    print("Evaluating on canonical batch...")
    payload = task.evaluate(model_fn)
    
    # Record benchmark
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    
    print(f"Payload: {payload}")
    print(f"Results written to: {run_dir}")