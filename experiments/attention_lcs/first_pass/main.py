import torch
import numpy as np

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"


class LCSAttentionModel(torch.nn.Module):
    """Simple cross-attention model for LCS alignment detection.

    Embeds both sequences, projects to Q/K/V, computes cross-attention
    from seq_a (queries) to seq_b (keys).
    """

    def __init__(self, vocab_size: int = 8, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Token embeddings (shared between seq_a and seq_b)
        self.embedding = torch.nn.Embedding(vocab_size, d_model)

        # Q projection for seq_a (queries)
        self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
        # K, V projections for seq_b (keys/values)
        self.k_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.v_proj = torch.nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=False)

        # Initialize with small weights
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)

    def forward(self, seq_a: torch.Tensor, seq_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_a: [batch, seq_len] int64
            seq_b: [batch, seq_len] int64
        Returns:
            attn_weights: [batch, n_heads, seq_len, seq_len] - attention from seq_a to seq_b
        """
        batch, seq_len = seq_a.shape

        # Embed both sequences
        emb_a = self.embedding(seq_a)  # [batch, seq_len, d_model]
        emb_b = self.embedding(seq_b)  # [batch, seq_len, d_model]

        # Project to Q, K, V
        q = self.q_proj(emb_a)  # [batch, seq_len, d_model]
        k = self.k_proj(emb_b)  # [batch, seq_len, d_model]
        v = self.v_proj(emb_b)  # [batch, seq_len, d_model]

        # Reshape for multi-head attention
        # [batch, seq_len, n_heads, head_dim] -> [batch, n_heads, seq_len, head_dim]
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # scores: [batch, n_heads, seq_len_q, seq_len_k]

        # Softmax over keys (last dimension)
        attn_weights = torch.softmax(scores, dim=-1)

        return attn_weights


def model_fn(seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
    """Model function compatible with task.py's ModelFn signature.

    Converts numpy arrays to torch tensors on GPU, runs the model,
    returns attention weights as numpy array.
    """
    # Convert to torch tensors on GPU
    seq_a_t = torch.as_tensor(seq_a, dtype=torch.long, device=DEVICE)
    seq_b_t = torch.as_tensor(seq_b, dtype=torch.long, device=DEVICE)

    # Run model
    model = LCSAttentionModel(vocab_size=8, d_model=64, n_heads=4).to(DEVICE)
    model.eval()

    with torch.no_grad():
        attn = model(seq_a_t, seq_b_t)  # [batch, n_heads, seq_len, seq_len]

    # Return as numpy array (float32)
    return attn.detach().cpu().numpy().astype(np.float32)


def main():
    task = load_task(__file__)
    payload = task.evaluate(model_fn)

    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    print("Benchmark payload:")
    print(f"  version: {payload['version']}")
    print(f"  config: {payload['config']}")
    print(f"  random_baseline_mass: {payload['random_baseline_mass']:.4f}")
    for rec in payload["sweep"]:
        print(f"  head {rec['head']}: mass={rec['lcs_attention_mass']:.4f}, "
              f"lift={rec['lcs_lift']:.4f}, n_queries={rec['n_query_positions']}")


if __name__ == "__main__":
    main()