import gradio as gr
import numpy as np
import json
import os
from pathlib import Path
from agentic.experiments import benchmark_panel
from dataclasses import dataclass, field

# Utility to load the latest run directory under results/
def find_latest_run(goal_dir: Path) -> Path | None:
    attempts_dir = goal_dir.parents[1]  # experiments/attention Boundary/pass_2
    runs = []
    for attempt_dir in attempts_dir.iterdir():
        if attempt_dir.is_dir():
            results_dir = attempt_dir / "results"
            if results_dir.exists():
                for run_dir in sorted(results_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                    if (run_dir / "benchmark.json").exists():
                        runs.append((run_dir.stat().st_mtime, run_dir))
    if not runs:
        return None
    return max(runs)[1]

# Load benchmark JSON from a specific run
def load_run_benchmark(run_dir: Path) -> dict:
    if not (run_dir / "benchmark.json").exists():
        print(f"[app.py] No benchmark.json in {run_dir}")
        return {}
    with open(run_dir / "benchmark.json") as f:
        return json.load(f)

# Load attention weights (NPY) from a run (exists if main.py saved it)
def load_run_attn(run_dir: Path) -> np.ndarray | None:
    attn_path = run_dir / "attn_weights.npy"
    if not attn_path.exists():
        print(f"[app.py] No attn_weights.npy in {run_dir}")
        return None
    return np.load(attn_path)

# Make a heatmap of a single head's attention matrix (batch-averaged)
def make_head_heatmap(attn: np.ndarray, head_idx: int) -> gr.Plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    head_attn = attn.mean(axis=0)[:, head_idx, :, :].mean(axis=0)  # (seq_len, seq_len)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(head_attn, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"Head {head_idx} Attention Heatmap")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.axvline(7.5, color="red", linestyle="--", alpha=0.6, label="DELIM (pos 8)")
    ax.axvline(16.5, color="orange", linestyle="--", alpha=0.6, label="EOS (pos 17)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig

# Bar chart of mean attention mass by region (within, delimiter, cross, EOS) for segA vs segB queries
def make_region_mass_chart(benchmark) -> gr.Plot:
    sweep = benchmark.get("sweep", [])
    if not sweep:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    regions = ["within_seg_attn", "delim_attn", "cross_seg_attn", "eos_attn"]
    region_labels = ["Within segment", "Delimiter", "Cross segment", "EOS"]
    x = np.arange(len(regions))
    width = 0.45

    # Build a small dictionary for legend handling
    for i, rec in enumerate(sweep):
        if "query_segment" not in rec:
            continue
        vals = [rec.get(r) for r in regions]
        offset = (i - 0.5) * width
        rects = ax.bar(x + offset, vals, width, label=rec["query_segment"], color=plt.cm.tab10(i % 10))
        rects[1].set_hatch('//')  # highlight delimiter

    ax.set_ylabel("Mean attention mass")
    ax.set_title("Attention Mass by Region")
    ax.set_xticks(x)
    ax.set_xticklabels(region_labels)
    ax.set_ylim(0, 1.05)
    handles, labels = ax.get_legend_handles_labels()
    rect = plt.Rectangle((0, 0), 1, 1, color='tab:gray', hatch='//', alpha=0.6)
    handles.append(rect)
    labels.append("Delimiter")
    ax.legend(handles, labels, title="Query Segment")
    plt.tight_layout()
    return fig

# Per-head sharpness bar chart
def make_head_sharpness_chart(benchmark) -> gr.Plot:
    sweep = benchmark.get("sweep", [])
    if not sweep:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, rec in enumerate(sweep):
        head_sharp = rec.get("head_sharpness", [])
        heads = list(range(len(head_sharp)))
        ax.bar(heads, head_sharp, alpha=0.7, label=rec.get("query_segment", ""), color=plt.cm.tab10(i % 10))
        ax.axhline(0.0, color="dimgray", linestyle="--", alpha=0.6)

    ax.set_xlabel("Head index")
    ax.set_ylabel("Boundary Sharpness (Within - max(Delimiter, Cross, EOS))")
    ax.set_title("Per-Head Sharpness")
    ax.legend()
    plt.tight_layout()
    return fig

# Demo tab builder — interactive selector to pick any run under the goal
def create_demo_tab():
    goal_dir = Path(__file__).parent.parent   # experiments/attention_boundary/pass_2
    latest_run = find_latest_run(goal_dir)
    # Keep a hidden state to store the current selected run path
    run_state = gr.State(latest_run)

    # UI elements
    with gr.Tab("Demo"):
        gr.Markdown("## Attention Boundary Demo\n"
                    "The model implements a **single engineered boundary-detector head**"
                    "while all other heads stay uniform (baseline).")
        with gr.Row():
            run_selector = gr.Dropdown(label="Select Run", choices=[], interactive=True)
            refresh_btn = gr.Button("-refresh runs", size="sm")
        with gr.Row():
            region_mass_chart = gr.Plot(label="Region Attention Mass")
            head_sharpness_chart = gr.Plot(label="Per-Head Sharpness")
        with gr.Row():
            head_selector = gr.Radio(
                choices=[str(h) for h in range(4)], label="Select Attention Head", value="0"
            )
            attn_heatmap = gr.Plot(label="Heatmap of Selected Head (averaged over batch)")
        with gr.Row():
            metrics_display = gr.JSON(label="Key Metrics", visible=False)

        # Helper: load selected run and return (run_path, benchmark, attn)
        def load_selected_run(selected_run):
            run_path = goal_dir / selected Run
            bench = load_run_benchmark(run_path)
            attn = load_run_attn(run_path)
            return bench, attn, bench.get("version", 0), bench.get("sweep", [{}])[0].get("seq_len", 18)

        # On run selector change
        def on_run_select(selected_run):
            bench, attn, ver, seq_len = load_selected_run(selected_run)
            # Update charts based on the new run
            region_fig = make_region_mass_chart(bench) if bench else None
            sharpness_fig = make_head_sharpness_chart(bench) if bench else None
            # Set head selector to head 0 (the engineered boundary head)
            head_sel = "0"
            # Generate heatmap for head 0 now
            heatmap_fig = make_head_heatmap(attn, int(head_sel)) if attn is not None else None
            return region_fig, sharpness_fig, head_sel, heatmap_fig, json.dumps(bench, indent=2), seq_len

        # On head selector change
        def on_head_select(selected_head, attn, bench, seq_len):
            if attn is None:
                return None
            try:
                head_idx = int(selected_head)
                fig = make_head_heatmap(attn, head_idx)
                return fig
            except Exception as e:
                print(f"[app.py] Heatmap error: {e}")
                return None

        # Initial load of latest run
        def init_latest():
            if not latest_run:
                return [], None, [], None
            bench, attn, ver, seq_len = load_selected_run(str(latest_run.name))
            all_runs = sorted(
                [r.relative_to(goal_dir) for r in goal_dir.rglob("**/results/*") if r.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            choices = [str(r) for r in all_runs]
            if not choices:
                return [], None, [], None
            # Return: runs Dropdown choices, latest run path, initial metrics JSON, sequence length
            return choices, str(latest_run.name), choices[0], bench, attn, seq_len

        # Populate runs list at start
        with gr.Block():
            refresh_btn.click(
                fn=lambda x: (x, x),  # dummy to trigger downstream logic
                inputs=[run_selector],
                outputs=[run_selector]
            ).then(
                fn=on_run_select,
                inputs=[run_selector],
                outputs=[
                    region_mass_chart,
                    head_sharpness_chart,
                    head_selector,
                    attn_heatmap,
                    metrics_display
                ]
            )

        # Run selector change handler
        head_selector.change(
            fn=on_head_select,
            inputs=[head_selector, run_state],
            outputs=[attn_heatmap]
        )

        # Run selector change handler
        run_selector.change(
            fn=on_run_select,
            inputs=[run_selector],
            outputs=[
                region_mass_chart,
                head_sharpness_chart,
                head_selector,
                attn_heatmap,
                metrics_display
            ]
        )

        demo.load(
            init_latest,
            outputs=[
                run_selector,
                run_state,
                run_selector,
                metrics_display,
                region_mass_chart,
                head_sharpness_chart,
                attn_heatmap
            ],
            cancels=[run_selector.change]
        )

# Create the full app (Demo tab + Benchmark tab)
def create_app():
    goal_dir = Path(__file__).parent.parent

    with gr.Blocks(title="attention Boundary - pass_2") as demo:
        with gr.Tabs():
            create_demo_tab()
            with gr.Tab("Benchmark"):
                benchmark_panel(str(goal_dir))

    return demo

demo = create_app()

if __name__ == "__main__":
    demo.launch()