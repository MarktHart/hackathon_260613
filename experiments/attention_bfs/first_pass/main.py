import torch
from torch.nn import Module, Linear
import numpy as np
from typing import Callable

from agentic.experiments import load_task, record_benchmark, results_dir

# GPU guarantee; no fallback.
DEVICE = "cuda"

# Model: single transformer block → one attention head with a 1-dim logit head.
class BFSAttention(Module):
    def __init__(self):
        super().__init__()
        self.token = torch.nn.Parameter(torch.empty(128, 64))
        np.random.seed(0)  # determinism for embedding
        self.token.data.copy_(torch.as_tensor(
            np.normal(size=(128, 64)),
            dtype=torch.float32,
            device=DEVICE
        ))

        # Attention: QK dot-product → logit per node.
        # We cheat a bit: make the head compute adjacency+embedding.
        self.Q_proj = Linear(64, 1)
        self.K_proj = Linear(64, 1)  # will be replaced by the adjacency matrix.
        self.V_proj = Linear(64, 1)  # projection to 1D logit.

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize as a near-identity transform.
        self.Q_proj.weight.data.normal_(0, 0.01)
        self.K_proj.weight.data.copy_(torch.zeros_like(self.K_proj.weight) + 1.0)
        self.V_proj.weight.data.normal_(0, 0.01)

        # Initialize biases to be learned.
        self.Q_proj.bias.data.zero_()
        self.V_proj.bias.data.zero_()

    def forward(self, embeds: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # embeds: (N_NODES, 64)
        # adjacency: (N_NODES, N_NODES)

        # Project queries.
        Q = self.Q_proj(embeds)  # (N_NODES, 1)

        # K is replaced by the adjacency matrix.
        K = adjacency.unsqueeze(1)  # (N_NODES, 1, N_NODES) → reshape to match projection shape.
        K = K.transpose(-1, -2)

        # Project values.
        V = embeds.unsqueeze(1)  # (N_NODES, 1, 64) → reshape for projection.
        V = self.V_proj(V)  # (N_NODES, 1, 1)

        # Attention.
        scores = Q @ K  # (N_NODES, N_NODES)
        attn = torch.softmax(scores, dim=-1)  # (N_NODES, N_NODES)
        logits = attn @ V  # (N_NODES, 1)

        return logits.squeeze(-1)  # (N_NODES,)

# Hand-written attention implementation, no training.
def attention_bfs(node_embeds: np.ndarray, adjacency: np.ndarray, frontier_mask: np.ndarray) -> np.ndarray:
    n_nodes = adjacency.shape[0]
    embeds_tensor = torch.as_tensor(node_embeds, dtype=torch.float32, device=DEVICE)  # (N_NODES, 64)
    adjacency_tensor = torch.as_tensor(adjacency, dtype=torch.float32, device=DEVICE)    # (N_NODES, N_NODES)

    # The QK term here is simply the adjacency matrix + a learned node embedding component.
    queries = torch.randn((n_nodes, 1), device=DEVICE)  # learn bias term in training.
    keys = adjacency_tensor  # direct copy of adjacency matrix.

    # Soft attention.
    logits = queries @ keys  # (N_NODES, 1) @ (N_NODES, N_NODES) = (N_NODES, N_NODES)

    # Softmax over the key dimension.
    attn = torch.softmax(logits, dim=-1)  # (N_NODES, N_NODES)

    # Value projection: add a learned bias per node.
    values = torch.randn((1, n_nodes), device=DEVICE)
    attn_logits = (attn @ values).squeeze(0)  # (N_NODES,)

    # Subtraction of frontier: logical AND-NOT.
    # This is the "not already reached" component.
    attn_logits[frontier_mask] = -1000.0  # Mask out frontier nodes.

    return attn_logits.detach().cpu().numpy()

# Main experiment entry point.
def main():
    task = load_task(__file__)

    # The synthetic batch is deterministic; `generate` returns a single instance.
    batch = task.generate(seed=42)

    # Convert to tensors for the GPU.
    embeds_tensors = [torch.as_tensor(g, dtype=torch.float32, device=DEVICE) for g in batch.node_embeds]
    adjacency_tensors = [torch.as_tensor(g, dtype=torch.float32, device=DEVICE) for g in batch.adjacencies]
    frontier_tensors = [torch.as_tensor(g, dtype=torch.bool, device=DEVICE) for g in batch.frontiers]

    # Compute logs across all seeds for each p in sweep.
    by_p: dict[float, dict] = {p: [] for p in batch.ps}
    base_by_p: dict[float, float] = {p: [] for p in batch.ps}

    for embeds, adjacency, frontier, label, p_val in zip(
        embeds_tensors, adjacency_tensors, frontier_tensors, batch.labels, batch.ps
    ):
        # --- Attempt: hand-written attention head ---
        attn_logits = attention_bfs(embeds.cpu().numpy(), adjacency.cpu().numpy(), frontier.cpu().numpy())
        attn = torch.softmax(torch.as_tensor(attn_logits, dtype=torch.float32, device=DEVICE), dim=0)
        attention_logits = attn.detach().cpu().numpy()

        # Compute metrics.
        sharpness = task._sharpness(attention_logits, label)
        # FPR and FNR are left out of the payload in this version.
        by_p[p_val].append({
            "bfs_sharpness": float(sharpness),
            "n_seeds": 1,
        })

        # --- Linear baseline: adjacency @ frontier ---
        base_score = np.tensordot(adjacency.cpu().numpy(), frontier.cpu().numpy(), axes=1)
        base_sharpness = task._sharpness(base_score, label)
        base_by_p[p_val].append(float(base_sharpness))

    # Format payload.
    sweep = [rec for val in by_p.values() for rec in val]
    linear_baseline = [rec for val in base_by_p.values() for rec in val]

    payload = {
        "version": 1,
        "model_name": "synthetic_attention_bfs",
        "n_nodes": 32,
        "d": 64,
        "canonical_p": 0.10,
        "p_sweep": [0.05, 0.10, 0.20, 0.30, 0.40],
        "sweep": [ {"p": p, "bfs_sharpness": sharpness, "n_seeds": 1} for p, sharpness in zip(batch.ps, [r["bfs_sharpness"] for r in sweep]) ],
        "linear_baseline": [ {"p": p, "bfs_sharpness": sharpness, "n_seeds": 1} for p, sharpness in zip(batch.ps, [r["linear_baseline_sharpness"] for r in linear_baseline]) ],
    }

    payload_dir = results_dir(__file__)
    record_benchmark(__file__, payload_dir, payload)
    print("Benchmark payload written to:", payload_dir)

if __name__ == "__main__":
    main()
