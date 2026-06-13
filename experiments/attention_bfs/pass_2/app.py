import gradio as gr
import numpy as np
import torch
from matplotlib import pyplot as plt

from agentic.experiments import benchmark_panel, load_task, results_dir

DEVICE = "cuda"

# Re-use the same model_fn logic as main.py for the demo.
def model_fn(adjacency: np.ndarray, source: int, hops: int) -> np.ndarray:
    n = adjacency.shape[0]
    adj_t = torch.as_tensor(adjacency, dtype=torch.float32, device=DEVICE)
    frontier = torch.zeros(n, device=DEVICE)
    frontier[source] = 1.0
    visited = frontier.clone()
    for _ in range(hops):
        scores = frontier @ adj_t
        next_frontier = torch.clamp(scores, 0.0, 1.0)
        visited = torch.maximum(visited, next_frontier)
        frontier = next_frontier
    return visited.detach().cpu().numpy()


def _bfs_truth(adj: np.ndarray, source: int, hops: int) -> np.ndarray:
    """Ground-truth reachability via standard BFS."""
    from collections import deque
    n = adj.shape[0]
    dist = np.full(n, np.inf)
    dist[source] = 0
    q = deque([source])
    while q:
        u = q.popleft()
        for v in np.nonzero(adj[u] > 0)[0]:
            if dist[v] == np.inf:
                dist[v] = dist[u] + 1
                q.append(int(v))
    return dist <= hops


def run_demo(p_val: float, seed: int, hops: int):
    """Generate one graph and visualise iterative propagation."""
    rng = np.random.RandomState(seed)
    n = 24
    adj = (rng.rand(n, n) < p_val).astype(float)
    np.fill_diagonal(adj, 0.0)
    source = int(rng.randint(n))

    # Run model at each hop to show propagation.
    hop_probs = []
    for h in range(1, hops + 1):
        hop_probs.append(model_fn(adj, source, h))

    gt = _bfs_truth(adj, source, hops)
    final_probs = hop_probs[-1]
    final_pred = final_probs >= 0.5

    # Plot: adjacency heatmap + propagation over hops + final comparison.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Adjacency matrix
    ax = axes[0, 0]
    im = ax.imshow(adj, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_title(f"Adjacency (p={p_val}, seed={seed})")
    ax.set_xlabel("Target node")
    ax.set_ylabel("Source node")
    ax.plot(source, source, "r*", markersize=15, label=f"Source {source}")
    ax.legend()

    # Propagation heatmap: rows = hops, cols = nodes, color = probability
    ax = axes[0, 1]
    prop_matrix = np.stack(hop_probs)  # (hops, n)
    im = ax.imshow(prop_matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_title("Reachability probability per hop")
    ax.set_xlabel("Node")
    ax.set_ylabel("Hop budget")
    ax.set_yticks(range(hops))
    ax.set_yticklabels([str(i + 1) for i in range(hops)])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Final probabilities vs ground truth
    ax = axes[1, 0]
    x = np.arange(n)
    ax.bar(x, final_probs, alpha=0.6, label="Model probability", color="steelblue")
    ax.axhline(0.5, color="gray", linestyle="--", label="Threshold 0.5")
    gt_idx = np.where(gt)[0]
    ax.scatter(gt_idx, final_probs[gt_idx], color="red", s=60, zorder=5, label="Ground-truth reachable")
    fp_idx = np.where(final_pred & ~gt)[0]
    if len(fp_idx):
        ax.scatter(fp_idx, final_probs[fp_idx], color="orange", s=60, marker="x", zorder=5, label="False positive")
    fn_idx = np.where(~final_pred & gt)[0]
    if len(fn_idx):
        ax.scatter(fn_idx, final_probs[fn_idx], color="red", s=60, marker="x", zorder=5, label="False negative")
    ax.set_title(f"Final prediction (hops={hops})")
    ax.set_xlabel("Node")
    ax.set_ylabel("Probability")
    ax.legend(fontsize=8)

    # Metrics text
    ax = axes[1, 1]
    ax.axis("off")
    tp = int(np.sum(final_pred & gt))
    fp = int(np.sum(final_pred & ~gt))
    fn = int(np.sum(~final_pred & gt))
    tn = int(np.sum(~final_pred & ~gt))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / n
    text = (
        f"Graph: n={n}, p={p_val}, source={source}, hops={hops}\n\n"
        f"TP={tp}  FP={fp}  FN={fn}  TN={tn}\n"
        f"Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}  Acc={acc:.3f}\n\n"
        f"Reachable nodes (GT): {np.sum(gt)}\n"
        f"Predicted reachable: {np.sum(final_pred)}"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    plt.tight_layout()
    return fig


def _load_latest_payload():
    """Load the most recent benchmark.json from this attempt's results dir."""
    import json
    from pathlib import Path
    run_dirs = sorted(results_dir(__file__).parent.glob("*"))
    if not run_dirs:
        return None
    latest = run_dirs[-1] / "benchmark.json"
    if latest.exists():
        with open(latest) as f:
            return json.load(f)
    return None


def plot_sweep():
    """Plot F1 vs hops for model vs baseline from the latest run."""
    payload = _load_latest_payload()
    if payload is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No benchmark run found.\nRun main.py first.", ha="center", va="center")
        ax.axis("off")
        return fig

    hops = [r["hops"] for r in payload["sweep"]]
    model_f1 = [r["model_f1"] for r in payload["sweep"]]
    base_f1 = [r["baseline_f1"] for r in payload["sweep"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hops, model_f1, "o-", label="Iterative attention (this attempt)", color="steelblue", linewidth=2)
    ax.plot(hops, base_f1, "s--", label="1-hop baseline", color="gray", linewidth=2)
    ax.set_xlabel("Hop budget h")
    ax.set_ylabel("Pooled F1")
    ax.set_title("Multi-hop reachability: model vs baseline")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# --- Gradio App ---
with gr.Blocks() as demo:
    gr.Markdown("# attention_bfs — pass_2: Iterative attention for multi-hop BFS")
    gr.Markdown(
        "Each attention step propagates the frontier one hop using the adjacency "
        "matrix as the attention pattern. Stacking `h` steps implements genuine "
        "multi-hop reachability, unlike a single attention layer which can only "
        "see 1-hop neighbors."
    )

    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                p_slider = gr.Slider(label="Edge probability p", minimum=0.05, maximum=0.40, value=0.10, step=0.05)
                seed_slider = gr.Slider(label="Random seed", minimum=0, maximum=100, value=0, step=1)
                hops_slider = gr.Slider(label="Hop budget h", minimum=1, maximum=5, value=5, step=1)
            vis_btn = gr.Button("Visualise")
            demo_plot = gr.Plot()

            vis_btn.click(
                fn=run_demo,
                inputs=[p_slider, seed_slider, hops_slider],
                outputs=demo_plot
            )
            # Also run on load with defaults
            demo.load(
                fn=run_demo,
                inputs=[p_slider, seed_slider, hops_slider],
                outputs=demo_plot
            )

        with gr.Tab("Benchmark"):
            # Leaderboard + history across all attempts in this goal.
            benchmark_panel("experiments/attention_bfs")

        with gr.Tab("Sweep Plot"):
            gr.Markdown("F1 vs hop budget from the latest run of this attempt.")
            sweep_plot = gr.Plot()
            refresh_btn = gr.Button("Refresh from latest run")
            refresh_btn.click(fn=plot_sweep, outputs=sweep_plot)
            demo.load(fn=plot_sweep, outputs=sweep_plot)

if __name__ == "__main__":
    demo.launch()