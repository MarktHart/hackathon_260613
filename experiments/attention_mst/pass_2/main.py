"""
attention_mst / pass_2  —  TRAINED attention denoiser  (+ causal ablation).

Why a trained, *structural* mechanism (not a per-edge denoiser):
  Kruskal depends ONLY on the ordering of the edge scores, so ANY monotone
  per-edge transform g(w_ij) of the noisy weights reproduces EXACTLY the
  strawman baseline (`-noisy_weights`). To denoise you must make an edge's
  score depend on the rest of the graph. We learn a permutation-equivariant
  self-attention network over the 12 nodes: each node is a token, the noisy
  weight matrix is injected as an additive attention bias (nodes attend along
  strong / low-weight edges), and an edge head reads the resulting node
  embeddings PLUS the raw weight (a skip) to score every pair. Trained on
  graphs from the same generator with seeds disjoint from the eval seed (42),
  BCE against the planted-MST edge labels.

  The raw-weight skip means the net can always fall back to the baseline, so
  the structural attention can only help. We prove it *uses* the attention with
  an ablation: zeroing the attention output collapses the curve toward the
  baseline.

Delta from base_model.py: one self-attention block (token = node) with a
weight-bias term in the logits, an MLP, then a symmetric edge head. Single
layer; all compute on CUDA.
"""

import os
import json
import math

import numpy as np
import torch
import torch.nn as nn

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU
torch.manual_seed(0)

task = load_task(__file__)
N = task.N_HEADS  # 12


class MSTNet(nn.Module):
    def __init__(self, d: int = 32):
        super().__init__()
        self.d = d
        self.node0 = nn.Parameter(torch.randn(d) * 0.1)
        self.rowfeat = nn.Linear(N, d)              # per-node edge-weight profile (sorted)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.tau = nn.Parameter(torch.tensor(1.0))  # temperature on the weight bias
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.ln2 = nn.LayerNorm(d)
        self.edge = nn.Sequential(nn.Linear(2 * d + 1, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, W, ablate: bool = False):  # W: (B, N, N) noisy weights
        B = W.shape[0]
        srow, _ = torch.sort(W, dim=-1)                       # permutation-invariant node feature
        x = self.node0[None, None, :] + self.rowfeat(srow)    # (B, N, d)

        Q, K, Vv = self.q(x), self.k(x), self.v(x)
        bias = -W / (self.tau.abs() + 1e-3)                   # attend along strong (low-weight) edges
        att = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.d) + bias
        att = torch.softmax(att, dim=-1)
        h = torch.matmul(att, Vv)
        if ablate:                                            # causal knock-out of the attention block
            h = torch.zeros_like(h)
        x = self.ln1(x + h)
        x = self.ln2(x + self.ff(x))

        hi = x[:, :, None, :].expand(B, N, N, self.d)
        hj = x[:, None, :, :].expand(B, N, N, self.d)
        feat = torch.cat([hi * hj, torch.abs(hi - hj), W[..., None]], dim=-1)
        s = self.edge(feat).squeeze(-1)                       # (B, N, N)
        s = 0.5 * (s + s.transpose(1, 2))                     # symmetric scores
        return s


def build_data(seeds):
    Ws, Ls = [], []
    for sd in seeds:
        for b in task.generate(seed=int(sd)):
            Ws.append(np.asarray(b.noisy_weights, dtype=np.float32))
            lab = np.zeros((N, N), dtype=np.float32)
            for u, v in b.planted_mst_edges:
                lab[int(u), int(v)] = 1.0
                lab[int(v), int(u)] = 1.0
            Ls.append(lab)
    W = torch.as_tensor(np.stack(Ws), device=DEVICE)
    L = torch.as_tensor(np.stack(Ls), device=DEVICE)
    return W, L


def train(steps: int = 600):
    net = MSTNet().to(DEVICE)
    W, L = build_data(range(0, 12))                # disjoint from eval seed 42
    iu = torch.triu_indices(N, N, 1, device=DEVICE)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0, device=DEVICE))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)

    M = W.shape[0]
    g = torch.Generator(device=DEVICE).manual_seed(0)
    net.train()
    for _ in range(steps):
        idx = torch.randint(0, M, (128,), generator=g, device=DEVICE)
        s = net(W[idx])
        loss = lossfn(s[:, iu[0], iu[1]], L[idx][:, iu[0], iu[1]])
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return net


def make_model_fn(net, ablate: bool = False):
    def model_fn(noisy_weights: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            w = torch.as_tensor(noisy_weights[None], dtype=torch.float32, device=DEVICE)
            s = net(w, ablate=ablate)[0]
            s.fill_diagonal_(0.0)
        return s.detach().cpu().numpy().astype(np.float64)
    return model_fn


if __name__ == "__main__":
    net = train()

    payload = task.evaluate(make_model_fn(net, ablate=False))
    ablate_payload = task.evaluate(make_model_fn(net, ablate=True))

    run_dir = results_dir(__file__)
    summary = {
        "noise_levels": payload["noise_levels"],
        "canonical_noise": payload["canonical_noise"],
        "method_f1": [r["edge_f1"] for r in payload["sweep"]],
        "baseline_f1": [r["edge_f1"] for r in payload["baseline"]],
        "ablate_f1": [r["edge_f1"] for r in ablate_payload["sweep"]],
        "method_auroc": [r["auroc"] for r in payload["sweep"]],
        "baseline_auroc": [r["auroc"] for r in payload["baseline"]],
        "ablate_auroc": [r["auroc"] for r in ablate_payload["sweep"]],
        "method_wratio": [r["weight_ratio"] for r in payload["sweep"]],
        "baseline_wratio": [r["weight_ratio"] for r in payload["baseline"]],
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    record_benchmark(__file__, run_dir, payload)

    nl = payload["noise_levels"]
    ci = nl.index(payload["canonical_noise"])
    print("noise  method  ablate  baseline")
    for i, x in enumerate(nl):
        print(f"{x:4.1f}   {summary['method_f1'][i]:.3f}   "
              f"{summary['ablate_f1'][i]:.3f}   {summary['baseline_f1'][i]:.3f}")
    print(f"\ncanonical lift over baseline: "
          f"{summary['method_f1'][ci] - summary['baseline_f1'][ci]:+.3f}")
    print(f"mst_recovery (mean F1): method={np.mean(summary['method_f1']):.3f} "
          f"ablate={np.mean(summary['ablate_f1']):.3f} "
          f"baseline={np.mean(summary['baseline_f1']):.3f}")
