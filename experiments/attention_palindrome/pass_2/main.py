import torch
import torch.nn as nn
import torch.nn.functional as F

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"
CANONICAL_SEED = 42   # matches task.py canonical seed
VOCAB = 8             # matches task.VOCAB
SEQ_LEN = 16          # matches task.SEQ_LEN

task = load_task(__file__)
run_dir = results_dir(__file__)


# ---- model definition -----------------------------------------------------
class PalindromeAttention(nn.Module):
    """A small, coherent single-head self-attention model with a palindrome
    readout. The intended approach (a trained attention head + MLP readout) is
    preserved, but the dimensions are made internally consistent so the matmuls
    actually compose.

    Shapes: tokens (B, L) -> embed (B, L, D) -> attention (B, L, D) -> mean-pool
    -> MLP -> scalar score (B,).
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.tok_emb = nn.Embedding(VOCAB, dim)
        self.pos_emb = nn.Embedding(SEQ_LEN, dim)

        # single-head self attention (q, k, v all D -> D)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # feed-forward
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        # readout: pooled hidden -> scalar palindrome score
        self.readout = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(tokens) + self.pos_emb(pos)        # (B, L, D)

        q = self.q_proj(x)                                  # (B, L, D)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = (q @ k.transpose(1, 2)) / (self.dim ** 0.5)  # (B, L, L)
        attn = F.softmax(scores, dim=-1)
        x = x + self.o_proj(attn @ v)                       # (B, L, D)
        x = x + self.ff(x)                                  # (B, L, D)

        pooled = x.mean(dim=1)                              # (B, D)
        return self.readout(pooled).squeeze(-1)            # (B,)


def train_model(model, n_steps: int = 1500):
    device = torch.device(DEVICE)
    model = model.to(device)

    batch = task.generate(CANONICAL_SEED)
    tokens = torch.as_tensor(batch.tokens, dtype=torch.int64, device=device)
    is_pal = torch.as_tensor(batch.is_palindrome, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.train()
    for step in range(n_steps):
        optimizer.zero_grad()
        logits = model(tokens)                              # (B,)
        loss = F.binary_cross_entropy_with_logits(logits, is_pal)
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")
    return model


# ---- run --------------------------------------------------------------------
def model_fn(batch):
    device = torch.device(DEVICE)
    model = train_model(PalindromeAttention(dim=64))
    tokens = torch.as_tensor(batch.tokens, dtype=torch.int64, device=device)
    model.eval()
    with torch.no_grad():
        scores = model(tokens).detach().cpu().numpy()
    return scores


payload = task.evaluate(model_fn)
record_benchmark(__file__, run_dir, payload)
print(f"Benchmark written to {run_dir / 'benchmark.json'}")
