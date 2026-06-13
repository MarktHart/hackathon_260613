"""attention_graph_color / pass_3

A genuine hand-built attention head for proper graph colourings, plus an
ablation that shows *which* part of the head does the work.

Key idea (and the difference from a naive adjacency mask):
The headline metric ``color_separation`` averages attention over **all** node
pairs of a colour relation, not just edges. So the right circuit must put mass
on *every* differently-coloured pair — edges AND non-edges — and starve
same-coloured pairs. The score is therefore driven by the **colour** signal:

    score_ij = w_color * (colour_i != colour_j)  +  w_adj * adj_ij        (i != j)
    attn     = softmax_j(score_ij)                                         (row-wise)

`w_color` produces the broad colour separation; `w_adj` adds a boost so that
differently-coloured *edges* out-attend differently-coloured *non-edges*
(the `edge_respect` metric). The colour term is NOT masked by adjacency — that
masking is exactly what made the previous attempt collapse to row-normalised
adjacency (the colour computation became a no-op under a proper colouring).

Everything runs in torch on CUDA. Hand-set weights, no training.

This file also records:
  * benchmark.json (full mechanism) via record_benchmark,
  * ablation.json  — color_separation / edge_respect for 4 variants,
  * extended_range.json — separation vs n over ~1.5 orders of magnitude,
  * samples.npz / samples_meta.json — a few graphs' adj/colors/attn for heatmaps.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a GPU; no CPU fallback.

# --- Hand-set circuit weights (no learning) ---
W_COLOR = 8.0   # broad colour separation (all pairs)
W_ADJ = 4.0     # edge boost (edge_respect)
TEMP = 1.0      # softmax temperature


# ---------------------------------------------------------------------------
# The attention head (parameterised so we can ablate it)
# ---------------------------------------------------------------------------
def make_head(w_color: float, w_adj: float, temp: float = TEMP):
    """Return a model_fn(adj, feats) -> (n, n) attention, computed on CUDA."""

    def model_fn(adj: np.ndarray, feats: np.ndarray) -> np.ndarray:
        n = adj.shape[0]
        k = feats.shape[1] - 1  # last column is normalised degree

        adj_t = torch.as_tensor(adj, dtype=torch.float32, device=DEVICE)
        colors = torch.as_tensor(feats[:, :k], dtype=torch.float32, device=DEVICE)

        # one-hot colour dot product: 1.0 if same colour, 0.0 if different.
        same = colors @ colors.T            # (n, n)
        diff = 1.0 - same                   # 1.0 iff colours differ

        # Hand-set additive score. Colour is the primary driver; edges add a boost.
        score = w_color * diff + w_adj * adj_t

        # Forbid self-attention.
        eye = torch.eye(n, dtype=torch.float32, device=DEVICE)
        score = score.masked_fill(eye.bool(), -1e9)

        # Genuine attention: row-wise softmax with temperature.
        attn = torch.softmax(score / temp, dim=1)
        attn = attn * (1.0 - eye)           # keep diagonal exactly zero

        return attn.detach().cpu().numpy().astype(np.float32)

    return model_fn


# ---------------------------------------------------------------------------
# Small self-contained generator for the extended operating-range sweep
# ---------------------------------------------------------------------------
def _gen_adj(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    upper = rng.random((n, n)) < p
    mask = np.triu(upper, k=1)
    adj = (mask | mask.T).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj


def _greedy_color(adj: np.ndarray):
    n = adj.shape[0]
    deg = adj.sum(1).astype(int)
    order = np.argsort(-deg)
    colors = np.full(n, -1, dtype=np.int64)
    for u in order:
        nb = {int(colors[v]) for v in np.where(adj[u] > 0)[0] if colors[v] != -1}
        c = 0
        while c in nb:
            c += 1
        colors[u] = c
    return colors, int(colors.max()) + 1


def _build_feats(colors: np.ndarray, k: int, adj: np.ndarray) -> np.ndarray:
    n = len(colors)
    f = np.zeros((n, k + 1), dtype=np.float32)
    f[np.arange(n), colors] = 1.0
    f[:, -1] = adj.sum(1) / max(1.0, n - 1)
    return f


def _color_separation(attn: np.ndarray, colors: np.ndarray) -> float:
    n = len(colors)
    triu = np.triu(np.ones((n, n), dtype=bool), k=1)
    eq = colors[:, None] == colors[None, :]
    same = eq & triu
    diff = (~eq) & triu
    s = float(attn[same].mean()) if same.any() else 0.0
    d = float(attn[diff].mean()) if diff.any() else 0.0
    return d - s


def _load_benchmark_module() -> object:
    bench_path = Path(__file__).resolve().parent.parent / "benchmark.py"
    spec = importlib.util.spec_from_file_location("agc_benchmark_p3", bench_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    task = load_task(__file__)
    benchmark = _load_benchmark_module()
    run_dir = results_dir(__file__)

    full_fn = make_head(W_COLOR, W_ADJ)

    # ---- Headline benchmark: the full mechanism ----
    payload = task.evaluate(full_fn)
    record_benchmark(__file__, run_dir, payload)
    full_metrics = benchmark.score(payload)

    # ---- Ablation: which part of the head does the work? ----
    variants = {
        "full (color+edge)": make_head(W_COLOR, W_ADJ),
        "color-only": make_head(W_COLOR, 0.0),
        "edge-only (~pass_2)": make_head(0.0, W_ADJ),
    }
    ablation = {}
    for name, fn in variants.items():
        m = benchmark.score(task.evaluate(fn))
        ablation[name] = {
            "color_separation_canonical": float(m["color_separation_canonical"]),
            "edge_respect_canonical": float(m["edge_respect_canonical"]),
            "color_separation_overall": float(m["color_separation_overall"]),
        }
    # uniform no-mechanism baseline (read straight from the task's baseline sweep)
    ablation["uniform baseline"] = {
        "color_separation_canonical": float(full_metrics["linear_baseline_color_separation"]),
        "edge_respect_canonical": 0.0,
        "color_separation_overall": float(full_metrics["linear_baseline_color_separation"]),
    }
    (run_dir / "ablation.json").write_text(json.dumps(ablation, indent=2))

    # ---- Operating range: separation vs n over ~1.5 orders of magnitude ----
    rng = np.random.default_rng(0)
    ext_ns = [20, 40, 80, 160, 320]
    extended = []
    for n in ext_ns:
        seps = []
        for _ in range(3):
            adj = _gen_adj(n, 0.2, rng)
            colors, k = _greedy_color(adj)
            feats = _build_feats(colors, k, adj)
            attn = full_fn(adj, feats)
            seps.append(_color_separation(attn, colors))
        extended.append({"n": n, "color_separation": float(np.mean(seps)),
                         "std": float(np.std(seps))})
    (run_dir / "extended_range.json").write_text(json.dumps(extended, indent=2))

    # ---- Sample graphs for heatmaps (canonical n=40, three densities) ----
    batch = task.generate(seed=0)
    sample_idxs = [15, 20, 25]  # n=40 slice spans 15..29; p=0.1/0.2/0.3 reps
    npz_data = {}
    meta = []
    for j, gi in enumerate(sample_idxs):
        adj = batch.adjacency[gi]
        colors = np.asarray(batch.colorings[gi])
        feats = batch.features[gi]
        attn = full_fn(adj, feats)
        npz_data[f"adj_{j}"] = adj.astype(np.float32)
        npz_data[f"colors_{j}"] = colors.astype(np.int64)
        npz_data[f"attn_{j}"] = attn.astype(np.float32)
        density = float(np.triu(adj, 1).sum() / max(1.0, adj.shape[0] * (adj.shape[0] - 1) / 2))
        meta.append({
            "key": j,
            "label": f"n=40, density={density:.2f}, k={int(colors.max()) + 1} colours",
            "n": int(adj.shape[0]),
            "color_separation": _color_separation(attn, colors),
        })
    np.savez(run_dir / "samples.npz", **npz_data)
    (run_dir / "samples_meta.json").write_text(json.dumps(meta, indent=2))

    print("[pass_3] color_separation_canonical =",
          round(full_metrics["color_separation_canonical"], 5),
          "| lift =", round(full_metrics["lift_over_linear_baseline"], 5))
    print("[pass_3] ablation:",
          {k: round(v["color_separation_canonical"], 4) for k, v in ablation.items()})


if __name__ == "__main__":
    main()
