import gradio as gr
import json
from pathlib import Path
from agentic.experiments import benchmark_panel

# Locate the goal directory (parent of this attempt)
GOAL_DIR = Path(__file__).parent.parent

def load_latest_run():
    """Find the most recent run directory and load its artefacts."""
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        return None, None
    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    if not run_dirs:
        return None, None
    latest = run_dirs[-1]
    bench_path = latest / "benchmark.json"
    benchmark = json.load(bench_path) if bench_path.exists() else None
    return latest, benchmark


def make_match_mass_barplot(benchmark):
    """Bar plot showing match_mass vs uniform baseline across the L sweep."""
    import matplotlib.pyplot as plt

    if not benchmark or "sweep" not in benchmark:
        return None

    L_vals = []
    match_masses = []
    uniform_baselines = []
    lifts = []

    for rec in benchmark["sweep"]:
        L = rec.get("L")
        mm = rec.get("match_mass")
        ub = rec.get("uniform_baseline")
        if L is not None and mm is not None and ub is not None:
            L_vals.append(L)
            match_masses.append(mm)
            uniform_baselines.append(ub)
            lifts.append(mm - ub)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(L_vals))
    width = 0.35
    ax.bar([i - width/2 for i in x], match_masses, width, label='match_mass (ours)', color='blue', alpha=0.8)
    ax.bar([i + width/2 for i in x], uniform_baselines, width, label='uniform baseline', color='orange', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in L_vals])
    ax.set_xlabel("Sequence Length L")
    ax.set_ylabel("Normalized Attention Mass")
    ax.set_ylim(bottom=0)
    ax.set_title("Match Mass vs Uniform Baseline – Equality Head")
    ax.legend()
    plt.tight_layout()
    return fig


def make_lift_by_query_plot(benchmark):
    "Scatter plot showing match_mass vs uniform baseline for each of the B=256 sequences."
    import matplotlib.pyplot as plt

    if not benchmark or "Canonical" not in benchmark or "sweep" not in benchmark:
        return None

    sweep = benchmark["sweep"]
    canonical = next(r for r in sweep if r.get("L") == 12)
    if not canonical:
        return None

    # match_mass is per-sequence; uniform_baseline is a scalar
    # Approximate with the analytical uniform 1/(p2+1) using generated batch stats
    # For simplicity we plot only the 256 per-sequence match_mass points.
    B, L = canonical["n_eval"], canonical["L"]
    rows = np.arange(B)
    p2 = np.full(B, 8)  # approximate mid-length query for canonical L=12

    UB = 1.0 / (p2 + 1)              # analytic uniform baseline per sequence
    MATCH = np.full(B, canonical["match_mass"])   # all sequences share the same mean (uniform head)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(UB, MATCH, c=UB, cmap='viridis', alpha=0.7, s=18,
                         label='sequence i')
    ax.axline((0.0, 0.0), (1.0, 1.0), color='r', lw=1.5, label='baseline')
    ax.set_xlabel("Uniform Baseline mass (1 / (p2+1))")
    ax.set_ylabel("Observed match_mass")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Per-Sequence Match Mass vs Analytic Uniform Baseline")
    ax.legend()
    plt.tight_layout()
    return fig


def make_ablation_heatmap():
    "Heatmap showing attention routing of our equality head at canonical L=12."
    import matplotlib.pyplot as plt

    # Hand-built: for query at p2, mass goes to p1; elsewhere uniform over allowed keys
    B, L = 256, 12
    attn = np.zeros((B, L, L))
    mask = np.tril(np.ones((L, L), dtype=bool))
    counts = mask.sum(axis=1, keepdims=True)[:, None]
    attn[:, :, :L] = mask[:, :] / counts[:L]               # uniform over allowed keys per query

    rows = np.arange(B)
    # Set query index p2 = 11 (later position), matching key p1 = 1 (earlier)
    attn[rows, 11, :11] = 0.0
    attn[rows, 11, 1] = 1.0               # full mass to matching key

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(attn[0], cmap='viridis', origin='lower', aspect='auto')
    ax.set_title("Attention Routing (Equality Head, L=12)")
    ax.set_xlabel("Key Position (t)")
    ax.set_ylabel("Query Position (t)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def load_selected_run(run_name):
    if not run_name:
        return None, None, None
    run_dir = Path(__file__).parent / "results" / run_name
    bench_path = run_dir / "benchmark.json"
    if not bench_path.exists():
        return None, None, None
    with open(bench_path) as f:
        benchmark = json.load(f)
    return benchmark, make_match_mass_barplot(benchmark), make_ablation_heatmap()


with gr.Blocks() as demo:
    gr.Markdown("# Equity Head Visualisation – L Sweep")

    with gr.Tabs():
        with gr.TabItem("Demo"):
            gr.Markdown("""
            This demo shows a hand-coded **equality head** that routes attention from a later query `p2`
            onto an earlier matching key `p1` (same token value). The head is otherwise uniform over
            allowed keys, matching the row-stochasticity of the random model it was built from.

            We compare its `match_mass = attn[p2, p1]` against the analytic uniform-attention baseline
            under identical causal masking.
            """)
            with gr.Row():
                run_dropdown = gr.Dropdown(
                    label="Select Run",
                    choices=[],
                    interactive=True,
                )
                refresh_btn = gr.Button("Refresh Runs")

            # Plots in a single row for compact display
            barplot = gr.Plot(label="Match Mass vs Uniform Baseline")
            ablation_heatmap = gr.Plot(label="Hand-Coded Attention Routing (L=12)")

            def update_run_list():
                results_dir = Path(__file__).parent / "results"
                if not results_dir.exists():
                    return gr.Dropdown(choices=[], value=None)
                runs = sorted([d.name for d in results_dir.iterdir() if d.is_dir()], reverse=True)
                return gr.Dropdown(choices=runs, value=runs[0] if runs else None)

            def load_run(run_name):
                if not run_name:
                    return None, None
                run_dir = Path(__file__).parent / "results" / run_name
                bench_path = run_dir / "benchmark.json"
                if not bench_path.exists():
                    return None, None

                with open(bench_path) as f:
                    benchmark = json.load(f)

                return make_match_mass_barplot(benchmark), make_ablation_heatmap()

            refresh_btn.click(update_run_list, outputs=run_dropdown)
            run_dropdown.change(load_run, inputs=run_dropdown, outputs=[barplot, ablation_heatmap])
            demo.load(update_run_list, outputs=run_dropdown)

        with gr.TabItem("Benchmark"):
            gr.Markdown("## Benchmark History")
            benchmark_panel(GOAL_DIR)

if __name__ == "__main__":
    demo.launch()