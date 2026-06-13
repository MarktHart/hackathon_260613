import json
import numpy as np
import gradio as gr
from pathlib import Path
from agentic.experiments import benchmark_panel


def load_latest_run(goal_dir: Path):
    """Find the most recent run directory under goal_dir/attempts/*/results/."""
    attempts_dir = goal_dir
    run_dirs = []
    for attempt_dir in attempts_dir.iterdir():
        if attempt_dir.is_dir():
            results_dir = attempt_dir / "results"
            if results_dir.exists():
                for run in results_dir.iterdir():
                    if run.is_dir() and (run / "benchmark.json").exists():
                        run_dirs.append((run.stat().st_mtime, run))
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda x: x[0], reverse=True)
    return run_dirs[0][1]


def load_run_data(run_dir: Path):
    """Load benchmark.json and any attention artefacts."""
    benchmark_path = run_dir / "benchmark.json"
    if not benchmark_path.exists():
        return None, None
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    # Try to load attention weights if saved
    attn_path = run_dir / "attention_weights.npy"
    attn = None
    if attn_path.exists():
        attn = np.load(attn_path)
    return benchmark, attn


def make_heatmap(attn, head_idx, title):
    """Create a heatmap figure for one head's attention matrix (averaged over batch)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Average over batch dimension
    attn_head = attn[:, head_idx, :, :].mean(axis=0)  # (seq_len, seq_len)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn_head, cmap="Blues", vmin=0, vmax=attn_head.max() or 1)
    ax.set_title(f"{title} - Head {head_idx}")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")

    # Add segment boundary lines
    seq_len = attn_head.shape[0]
    delim_pos = 8
    segA_end = 8
    segB_end = 17
    ax.axvline(delim_pos - 0.5, color="red", linestyle="--", alpha=0.7, label="DELIM")
    ax.axhline(delim_pos - 0.5, color="red", linestyle="--", alpha=0.7)
    ax.axvline(segB_end - 0.5, color="orange", linestyle="--", alpha=0.7, label="EOS")
    ax.axhline(segB_end - 0.5, color="orange", linestyle="--", alpha=0.7)

    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def make_region_bar_chart(benchmark):
    """Bar chart showing region attention masses for segA and segB queries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = benchmark.get("sweep", [])
    if not sweep:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    regions = ["within_seg_attn", "delim_attn", "cross_seg_attn", "eos_attn"]
    region_labels = ["Within segment", "Delimiter", "Cross segment", "EOS"]
    x = np.arange(len(regions))
    width = 0.35

    for i, rec in enumerate(sweep):
        if "query_segment" not in rec:
            continue
        vals = [rec.get(r, 0) for r in regions]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=f"Queries: {rec['query_segment']}")

    ax.set_ylabel("Attention mass (mean over batch, heads, queries)")
    ax.set_title("Attention mass by region")
    ax.set_xticks(x)
    ax.set_xticklabels(region_labels)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig


def make_sharpness_chart(benchmark):
    """Per-head sharpness for each segment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = benchmark.get("sweep", [])
    if not sweep:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, rec in enumerate(sweep):
        if "query_segment" not in rec:
            continue
        ax = axes[i]
        head_sharp = rec.get("head_sharpness", [])
        heads = list(range(len(head_sharp)))
        ax.bar(heads, head_sharp, color="steelblue")
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_title(f"{rec['query_segment']} - Per-head sharpness")
        ax.set_xlabel("Head")
        ax.set_ylabel("within - max(delim, cross, EOS)")
        ax.set_xticks(heads)

    plt.tight_layout()
    return fig


def create_demo_tab():
    """Build the Demo tab content."""
    with gr.Tab("Demo"):
        gr.Markdown("""
        # Attention Boundary Demo

        This attempt uses a hand-built attention pattern that explicitly respects segment boundaries.
        The model concentrates query attention within the query's own segment (A or B),
        with minimal leakage to the delimiter, the other segment, or EOS.
        """)

        with gr.Row():
            run_dropdown = gr.Dropdown(label="Select run", choices=[], interactive=True)
            refresh_btn = gr.Button("Refresh runs", size="sm")

        with gr.Row():
            benchmark_json = gr.JSON(label="Benchmark metrics", visible=False)

        with gr.Row():
            region_chart = gr.Plot(label="Region attention masses")
            sharpness_chart = gr.Plot(label="Per-head sharpness")

        with gr.Row():
            head_slider = gr.Slider(0, 3, value=0, step=1, label="Head index")
            with gr.Column():
                attn_heatmap = gr.Plot(label="Attention heatmap (batch-averaged)")

        def refresh_runs():
            goal_dir = Path(__file__).parent.parent
            run_dir = load_latest_run(goal_dir)
            if run_dir is None:
                return gr.Dropdown(choices=[], value=None), None
            # List all runs
            all_runs = []
            for attempt_dir in goal_dir.iterdir():
                if attempt_dir.is_dir():
                    results_dir = attempt_dir / "results"
                    if results_dir.exists():
                        for run in sorted(results_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                            if run.is_dir() and (run / "benchmark.json").exists():
                                rel = run.relative_to(goal_dir)
                                all_runs.append((str(rel), run))
            choices = [label for label, _ in all_runs]
            return gr.Dropdown(choices=choices, value=choices[0] if choices else None), all_runs[0][1] if all_runs else None

        def on_run_select(selected_label, all_runs_data=None):
            # We need to find the run path from the label
            goal_dir = Path(__file__).parent.parent
            run_path = goal_dir / selected_label
            benchmark, attn = load_run_data(run_path)
            if benchmark is None:
                return None, None, None, None
            region_fig = make_region_bar_chart(benchmark)
            sharp_fig = make_sharpness_chart(benchmark)
            heatmap_fig = None
            if attn is not None:
                heatmap_fig = make_heatmap(attn, 0, "Attention")
            return benchmark, region_fig, sharp_fig, heatmap_fig

        def on_head_change(head_idx, run_dir_path):
            if run_dir_path is None:
                return None
            goal_dir = Path(__file__).parent.parent
            run_path = goal_dir / run_dir_path
            _, attn = load_run_data(run_path)
            if attn is None:
                return None
            return make_heatmap(attn, int(head_idx), "Attention")

        # We need to store the run path for head_slider callback
        run_path_state = gr.State(value=None)

        refresh_btn.click(
            fn=refresh_runs,
            outputs=[run_dropdown, run_path_state]
        )

        run_dropdown.change(
            fn=lambda label: (label, label),
            inputs=[run_dropdown],
            outputs=[run_path_state, run_path_state]  # hack to trigger load
        ).then(
            fn=on_run_select,
            inputs=[run_dropdown, run_path_state],
            outputs=[benchmark_json, region_chart, sharpness_chart, attn_heatmap]
        )

        head_slider.change(
            fn=on_head_change,
            inputs=[head_slider, run_path_state],
            outputs=[attn_heatmap]
        )

        # Initial load
        demo.load(
            fn=refresh_runs,
            outputs=[run_dropdown, run_path_state]
        ).then(
            fn=lambda label, path: on_run_select(label, path) if label else (None, None, None, None),
            inputs=[run_dropdown, run_path_state],
            outputs=[benchmark_json, region_chart, sharpness_chart, attn_heatmap]
        )


def create_app():
    goal_dir = Path(__file__).parent.parent

    with gr.Blocks(title="attention_boundary - first_pass") as demo:
        with gr.Tabs():
            create_demo_tab()
            with gr.Tab("Benchmark"):
                benchmark_panel(str(goal_dir))

    return demo


demo = create_app()

if __name__ == "__main__":
    demo.launch()