import gradio as gr
import numpy as np
import json
from pathlib import Path
from agentic.experiments import benchmark_panel

# Find the latest run directory
def _get_latest_run_dir():
    base = Path(__file__).parent / "results"
    if not base.exists():
        return None
    runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    return runs[0] if runs else None


def _load_run_data(run_dir: Path):
    """Load all saved numpy arrays and benchmark.json from a run directory."""
    if run_dir is None:
        return None
    
    data = {}
    # Load benchmark.json
    bench_files = list(run_dir.glob("benchmark.json")) + list(run_dir.glob("benchmark_*.json"))
    if bench_files:
        with open(bench_files[0]) as f:
            data["benchmark"] = json.load(f)
    
    # Load numpy arrays
    for name in ["attn_weights", "Q", "K", "V", "gt_out", "sweep_preds", "sweep_gt", "sweep_attn"]:
        fpath = run_dir / f"{name}.npy"
        if fpath.exists():
            data[name] = np.load(fpath, allow_pickle=True)
            if name in ["sweep_preds", "sweep_gt", "sweep_attn"] and data[name].dtype == object:
                data[name] = data[name].item()
    
    return data


def _make_heatmap(matrix, title="", cmap="viridis", vmin=None, vmax=None):
    """Create a matplotlib heatmap figure for a 2D matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def _make_error_heatmap(pred, gt, title=""):
    """Create error heatmap (|pred - gt|) averaged over batch and heads."""
    # pred, gt: (B, H, S, D)
    error = np.abs(pred - gt)  # (B, H, S, D)
    error_mean = error.mean(axis=(0, 1, 3))  # (S,) mean over batch, heads, d_head
    # Or show per-token error
    error_per_token = error.mean(axis=(0, 1, 3))  # (S,)
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(error_per_token)), error_per_token)
    ax.set_title(title)
    ax.set_xlabel("Query position")
    ax.set_ylabel("Mean absolute error")
    plt.tight_layout()
    return fig


def _make_sweep_plot(benchmark_data):
    """Plot fidelity metrics across sequence lengths."""
    if not benchmark_data or "sweep" not in benchmark_data:
        return None
    
    sweep = benchmark_data["sweep"]
    seq_lens = [s["seq_len"] for s in sweep]
    mse_vals = [s["mse"] for s in sweep]
    cos_vals = [s["cos_sim"] for s in sweep]
    rel_vals = [s["rel_error"] for s in sweep]
    base_vals = [s["baseline_mse"] for s in sweep]
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # MSE
    axes[0, 0].plot(seq_lens, mse_vals, 'o-', label="Model MSE", color="blue")
    axes[0, 0].plot(seq_lens, base_vals, 's--', label="Baseline MSE (uniform)", color="red")
    axes[0, 0].set_xlabel("Sequence length")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].set_title("MSE vs Sequence Length")
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cosine similarity
    axes[0, 1].plot(seq_lens, cos_vals, 'o-', color="green")
    axes[0, 1].set_xlabel("Sequence length")
    axes[0, 1].set_ylabel("Cosine similarity")
    axes[0, 1].set_title("Cosine Similarity vs Sequence Length")
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Relative error
    axes[1, 0].plot(seq_lens, rel_vals, 'o-', color="orange")
    axes[1, 0].set_xlabel("Sequence length")
    axes[1, 0].set_ylabel("Relative Frobenius error")
    axes[1, 0].set_title("Relative Error vs Sequence Length")
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Fidelity (1 - MSE/baseline)
    fidelity = [1 - m/b if b > 0 else 0 for m, b in zip(mse_vals, base_vals)]
    axes[1, 1].plot(seq_lens, fidelity, 'o-', color="purple")
    axes[1, 1].set_xlabel("Sequence length")
    axes[1, 1].set_ylabel("Fidelity (1 - MSE/baseline)")
    axes[1, 1].set_title("Attention Fidelity vs Sequence Length")
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def _make_attention_comparison(attn_weights, head_idx=0, batch_idx=0):
    """Show attention weights for a specific head."""
    # attn_weights: (B, H, S, S)
    if attn_weights.ndim == 4:
        A = attn_weights[batch_idx, head_idx]
    else:
        A = attn_weights
    return _make_heatmap(A, f"Attention Weights (Head {head_idx}, Batch {batch_idx})", cmap="Blues")


def _make_pred_vs_gt_heatmap(pred, gt, head_idx=0, batch_idx=0, d_idx=0):
    """Show pred vs gt for a specific head and feature dimension."""
    # pred, gt: (B, H, S, D)
    if pred.ndim == 4:
        p = pred[batch_idx, head_idx, :, d_idx]
        g = gt[batch_idx, head_idx, :, d_idx]
    else:
        p = pred[:, d_idx]
        g = gt[:, d_idx]
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    
    axes[0].plot(p, 'o-', label="Prediction", alpha=0.8)
    axes[0].plot(g, 's-', label="Ground Truth", alpha=0.8)
    axes[0].set_title(f"Head {head_idx}, Dim {d_idx}")
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(p - g, 'o-', color="red", alpha=0.8)
    axes[1].set_title("Difference (Pred - GT)")
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Error")
    axes[1].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[2].scatter(g, p, alpha=0.5, s=10)
    lim = max(np.abs(g).max(), np.abs(p).max()) * 1.1
    axes[2].plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5)
    axes[2].set_xlabel("Ground Truth")
    axes[2].set_ylabel("Prediction")
    axes[2].set_title("Pred vs GT Scatter")
    axes[2].set_aspect("equal")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def _make_per_head_metrics(pred, gt):
    """Compute per-head metrics."""
    # pred, gt: (B, H, S, D)
    B, H, S, D = pred.shape
    metrics = []
    for h in range(H):
        p = pred[:, h].reshape(-1, D)
        g = gt[:, h].reshape(-1, D)
        mse = np.mean((p - g) ** 2)
        # Cosine similarity
        p_n = np.linalg.norm(p, axis=-1)
        g_n = np.linalg.norm(g, axis=-1)
        dots = np.sum(p * g, axis=-1)
        mask = (p_n > 0) & (g_n > 0)
        cos = np.zeros_like(dots)
        cos[mask] = dots[mask] / (p_n[mask] * g_n[mask])
        cos_sim = np.mean(cos)
        metrics.append({"head": h, "mse": mse, "cos_sim": cos_sim})
    return metrics


with gr.Blocks(css="""
    .info-footer {font-size: 0.75rem; color: #666;}
    .metric-box {padding: 10px; border-radius: 5px; background: #f5f5f5; margin: 5px 0;}
""") as demo:
    gr.Markdown("# attention_dot_product / pass_4 Demo")
    gr.Markdown("**Scaled dot-product attention fidelity across sequence lengths.** "
                "`softmax(QKᵀ/√d_head) · V` implemented in PyTorch on CUDA.")
    
    with gr.Tabs():
        with gr.TabItem("Overview"):
            with gr.Row():
                run_dropdown = gr.Dropdown(
                    label="Select run",
                    choices=[],
                    value=None,
                    interactive=True,
                )
                refresh_btn = gr.Button("Refresh runs", variant="secondary")
            
            with gr.Row():
                fidelity_metric = gr.Number(label="Attention Fidelity (sweep avg)", precision=6)
                canonical_cos = gr.Number(label="Cosine Similarity (seq_len=32)", precision=6)
                canonical_mse = gr.Number(label="MSE (seq_len=32)", precision=6)
                worst_cos = gr.Number(label="Worst Cosine (sweep)", precision=6)
            
            sweep_plot = gr.Plot(label="Sweep Metrics")
            
            gr.Markdown("### Canonical Condition (seq_len=32) Details")
            with gr.Row():
                head_selector = gr.Dropdown(
                    label="Attention Head",
                    choices=[0, 1, 2, 3],
                    value=0,
                )
                batch_selector = gr.Dropdown(
                    label="Batch Index",
                    choices=[0, 1, 2, 3, 4, 5, 6, 7],
                    value=0,
                )
                dim_selector = gr.Dropdown(
                    label="Feature Dimension",
                    choices=list(range(16)),
                    value=0,
                )
            
            with gr.Row():
                attn_heatmap = gr.Plot(label="Attention Weights (Query × Key)")
                pred_gt_plot = gr.Plot(label="Prediction vs Ground Truth")
            
            per_head_table = gr.Dataframe(
                label="Per-Head Metrics (seq_len=32)",
                headers=["Head", "MSE", "Cosine Similarity"],
                datatype=["number", "number", "number"],
            )
        
        with gr.TabItem("Sequence Length Sweep"):
            gr.Markdown("Compare model attention output vs ground truth at each sequence length.")
            with gr.Row():
                seqlen_selector = gr.Dropdown(
                    label="Sequence Length",
                    choices=[8, 16, 32, 64, 128],
                    value=32,
                )
                seqlen_head = gr.Dropdown(label="Head", choices=[0, 1, 2, 3], value=0)
                seqlen_batch = gr.Dropdown(label="Batch", choices=list(range(8)), value=0)
                seqlen_dim = gr.Dropdown(label="Feature Dim", choices=list(range(16)), value=0)
            
            seqlen_pred_gt = gr.Plot(label="Pred vs GT at Selected Length")
            seqlen_attn = gr.Plot(label="Attention Weights at Selected Length")
            seqlen_error = gr.Plot(label="Per-Token Error at Selected Length")
        
        with gr.TabItem("Benchmark History"):
            benchmark_panel("experiments/attention_dot_product")
    
    # --- Callbacks ---
    def _list_runs():
        base = Path(__file__).parent / "results"
        if not base.exists():
            return [], None
        runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
        choices = [p.name for p in runs]
        return choices, choices[0] if choices else None
    
    def _on_run_select(run_name):
        if not run_name:
            return [None]*8
        run_dir = Path(__file__).parent / "results" / run_name
        data = _load_run_data(run_dir)
        if not data:
            return [None]*8
        
        bench = data.get("benchmark", {})
        sweep = bench.get("sweep", [])
        
        # Overview metrics
        fidelity = bench.get("attention_fidelity", 0)
        canon = next((s for s in sweep if s["seq_len"] == 32), sweep[2] if len(sweep) > 2 else sweep[-1])
        canon_cos = canon.get("cos_sim", 0)
        canon_mse = canon.get("mse", 0)
        worst_cos = min(s.get("cos_sim", 1) for s in sweep) if sweep else 0
        
        # Sweep plot
        sweep_fig = _make_sweep_plot(bench)
        
        # Canonical data
        attn_weights = data.get("attn_weights")
        gt_out = data.get("gt_out")
        sweep_preds = data.get("sweep_preds", {})
        sweep_gt = data.get("sweep_gt", {})
        sweep_attn = data.get("sweep_attn", {})
        
        # Default head 0, batch 0, dim 0
        attn_fig = _make_attention_comparison(attn_weights, 0, 0) if attn_weights is not None else None
        
        pred_canon = sweep_preds.get(32)
        gt_canon = sweep_gt.get(32)
        pred_gt_fig = _make_pred_vs_gt_heatmap(pred_canon, gt_canon, 0, 0, 0) if pred_canon is not None and gt_canon is not None else None
        
        # Per-head metrics
        per_head = _make_per_head_metrics(pred_canon, gt_canon) if pred_canon is not None and gt_canon is not None else []
        per_head_rows = [[m["head"], m["mse"], m["cos_sim"]] for m in per_head]
        
        return [fidelity, canon_cos, canon_mse, worst_cos, sweep_fig, attn_fig, pred_gt_fig, per_head_rows]
    
    def _on_canonical_params_change(head, batch, dim, run_name):
        if not run_name:
            return None, None, None
        run_dir = Path(__file__).parent / "results" / run_name
        data = _load_run_data(run_dir)
        if not data:
            return None, None, None
        
        attn_weights = data.get("attn_weights")
        sweep_preds = data.get("sweep_preds", {})
        sweep_gt = data.get("sweep_gt", {})
        pred_canon = sweep_preds.get(32)
        gt_canon = sweep_gt.get(32)
        
        attn_fig = _make_attention_comparison(attn_weights, head, batch) if attn_weights is not None else None
        pred_gt_fig = _make_pred_vs_gt_heatmap(pred_canon, gt_canon, head, batch, dim) if pred_canon is not None and gt_canon is not None else None
        
        per_head = _make_per_head_metrics(pred_canon, gt_canon) if pred_canon is not None and gt_canon is not None else []
        per_head_rows = [[m["head"], m["mse"], m["cos_sim"]] for m in per_head]
        
        return attn_fig, pred_gt_fig, per_head_rows
    
    def _on_seqlen_change(seq_len, head, batch, dim, run_name):
        if not run_name:
            return None, None, None
        run_dir = Path(__file__).parent / "results" / run_name
        data = _load_run_data(run_dir)
        if not data:
            return None, None, None
        
        sweep_preds = data.get("sweep_preds", {})
        sweep_gt = data.get("sweep_gt", {})
        sweep_attn = data.get("sweep_attn", {})
        
        pred = sweep_preds.get(seq_len)
        gt = sweep_gt.get(seq_len)
        attn = sweep_attn.get(seq_len)
        
        pred_gt_fig = _make_pred_vs_gt_heatmap(pred, gt, head, batch, dim) if pred is not None and gt is not None else None
        attn_fig = _make_attention_comparison(attn, head, batch) if attn is not None else None
        
        # Error heatmap
        error_fig = None
        if pred is not None and gt is not None:
            # Average over feature dims for this head/batch
            p = pred[batch, head].mean(axis=-1)  # (S,)
            g = gt[batch, head].mean(axis=-1)    # (S,)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(range(len(p)), np.abs(p - g), alpha=0.7, color="red")
            ax.set_title(f"Per-Token Mean Abs Error (Head {head}, Batch {batch})")
            ax.set_xlabel("Query Position")
            ax.set_ylabel("Mean |Pred - GT|")
            plt.tight_layout()
            error_fig = fig
        
        return pred_gt_fig, attn_fig, error_fig
    
    # Wire up callbacks INSIDE the Blocks context
    refresh_btn.click(
        fn=_list_runs,
        inputs=[],
        outputs=[run_dropdown],
    ).then(
        fn=lambda choices, val: gr.update(choices=choices, value=val),
        inputs=[run_dropdown, gr.State()],
        outputs=[run_dropdown],
    )
    
    # Initial load
    demo.load(
        fn=_list_runs,
        inputs=[],
        outputs=[run_dropdown],
    ).then(
        fn=lambda choices: gr.update(choices=choices, value=choices[0] if choices else None),
        inputs=[run_dropdown],
        outputs=[run_dropdown],
    ).then(
        fn=_on_run_select,
        inputs=[run_dropdown],
        outputs=[fidelity_metric, canonical_cos, canonical_mse, worst_cos, sweep_plot, attn_heatmap, pred_gt_plot, per_head_table],
    )
    
    # Run selection change
    run_dropdown.change(
        fn=_on_run_select,
        inputs=[run_dropdown],
        outputs=[fidelity_metric, canonical_cos, canonical_mse, worst_cos, sweep_plot, attn_heatmap, pred_gt_plot, per_head_table],
    )
    
    # Canonical parameter changes
    for comp in [head_selector, batch_selector, dim_selector]:
        comp.change(
            fn=_on_canonical_params_change,
            inputs=[head_selector, batch_selector, dim_selector, run_dropdown],
            outputs=[attn_heatmap, pred_gt_plot, per_head_table],
        )
    
    # Sweep tab parameter changes
    for comp in [seqlen_selector, seqlen_head, seqlen_batch, seqlen_dim]:
        comp.change(
            fn=_on_seqlen_change,
            inputs=[seqlen_selector, seqlen_head, seqlen_batch, seqlen_dim, run_dropdown],
            outputs=[seqlen_pred_gt, seqlen_attn, seqlen_error],
        )
    
    # Initial load for sweep tab (triggered by run selection)
    run_dropdown.change(
        fn=_on_seqlen_change,
        inputs=[seqlen_selector, seqlen_head, seqlen_batch, seqlen_dim, run_dropdown],
        outputs=[seqlen_pred_gt, seqlen_attn, seqlen_error],
    )


if __name__ == "__main__":
    demo.launch()
