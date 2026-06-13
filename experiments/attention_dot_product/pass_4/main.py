import numpy as np
import torch
import json
from pathlib import Path

from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

DEVICE = "cuda"


def model_fn(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Scaled dot-product attention on the GPU.

    Q, K, V : (batch, n_heads, seq_len, d_head)
    Returns: (batch, n_heads, seq_len, d_head)
    """
    Qt = torch.from_numpy(Q).float().to(DEVICE)
    Kt = torch.from_numpy(K).float().to(DEVICE)
    Vt = torch.from_numpy(V).float().to(DEVICE)

    d_head = Qt.shape[-1]
    scale = 1.0 / torch.sqrt(torch.tensor(d_head, device=DEVICE, dtype=torch.float32))
    # (B, H, S, S) = (B, H, S, D) @ (B, H, D, S)
    scores = torch.einsum("bhsd,bhtd->bhst", Qt, Kt) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhst,bhtd->bhsd", attn, Vt)  # (B, H, S, d_head)
    return out.detach().cpu().numpy()


def compute_attention_weights(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Compute attention weight matrix for visualization (softmax(QK^T/sqrt(d)))."""
    Qt = torch.from_numpy(Q).float().to(DEVICE)
    Kt = torch.from_numpy(K).float().to(DEVICE)
    d_head = Qt.shape[-1]
    scale = 1.0 / torch.sqrt(torch.tensor(d_head, device=DEVICE, dtype=torch.float32))
    scores = torch.einsum("bhsd,bhtd->bhst", Qt, Kt) * scale
    attn = torch.softmax(scores, dim=-1)
    return attn.detach().cpu().numpy()


if __name__ == "__main__":
    # Run evaluation and get payload
    payload = task.evaluate(model_fn)
    
    # Save attention weights for the canonical batch for visualization
    run_dir = results_dir(__file__)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate canonical batch and compute attention weights
    canonical_batch = task.generate(seed=0)
    attn_weights = compute_attention_weights(canonical_batch.Q, canonical_batch.K)
    
    # Save attention weights and batch data for the app
    np.save(run_dir / "attn_weights.npy", attn_weights)  # (B, H, S, S)
    np.save(run_dir / "Q.npy", canonical_batch.Q)
    np.save(run_dir / "K.npy", canonical_batch.K)
    np.save(run_dir / "V.npy", canonical_batch.V)
    np.save(run_dir / "gt_out.npy", canonical_batch.gt_out)
    
    # Also compute model predictions for all sweep lengths for comparison viz
    sweep_preds = {}
    sweep_gt = {}
    sweep_attn = {}
    for seq_len in task.config()["seq_len_sweep"]:
        batch = task._generate_seq(seq_len, seed=0)  # internal but we need it
        pred = model_fn(batch.Q, batch.K, batch.V)
        attn = compute_attention_weights(batch.Q, batch.K)
        sweep_preds[seq_len] = pred
        sweep_gt[seq_len] = batch.gt_out
        sweep_attn[seq_len] = attn
    
    np.save(run_dir / "sweep_preds.npy", sweep_preds)
    np.save(run_dir / "sweep_gt.npy", sweep_gt)
    np.save(run_dir / "sweep_attn.npy", sweep_attn)
    
    # Record benchmark
    record_benchmark(__file__, run_dir, payload)
    print(f"Benchmark recorded to {run_dir}/benchmark.json")
    print(f"Attention fidelity: {payload['sweep'][2]['cos_sim']:.6f} (canonical seq_len=32)")
