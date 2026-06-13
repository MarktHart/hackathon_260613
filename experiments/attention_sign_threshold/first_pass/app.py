import gradio as gr
import numpy as np
from pathlib import Path
from agentic.experiments import benchmark_panel, load_task, results_dir

# Load the goal's task to access canonical config and model_fn for demo
task = load_task(__file__)

# Recreate the same model_fn used in main.py for interactive demo
BETA = 20.0


def model_fn(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    cosine = np.sum(q * k, axis=1)
    return np.tanh(BETA * cosine)


def run_demo(cosine: float, n_pairs: int, beta: float, seed: int):
    """Generate pairs at a specific cosine and show score distribution."""
    rng = np.random.default_rng(seed)
    dim = task.CANONICAL_DIM

    # Generate q, k at exact cosine (mirrors task._generate_pairs_for_cosine)
    q = rng.normal(size=(n_pairs, dim))
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12

    sin_val = np.sqrt(max(0.0, 1.0 - cosine * cosine))
    ortho = rng.normal(size=(n_pairs, dim))
    ortho -= np.sum(ortho * q, axis=1, keepdims=True) * q
    ortho /= np.linalg.norm(ortho, axis=1, keepdims=True) + 1e-12

    k = cosine * q + sin_val * ortho
    k /= np.linalg.norm(k, axis=1, keepdims=True) + 1e-12

    # Compute scores with adjustable beta
    scores = np.tanh(beta * np.sum(q * k, axis=1))

    target_sign = 1 if cosine > 0 else (-1 if cosine < 0 else 0)
    sign_match = float(np.mean(np.sign(scores) == target_sign)) if target_sign != 0 else 0.0

    # Histogram data
    hist, bins = np.histogram(scores, bins=40, range=(-1.1, 1.1))

    return {
        "cosine": cosine,
        "target_sign": target_sign,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "sign_match": sign_match,
        "hist_counts": hist.tolist(),
        "hist_bins": bins.tolist(),
        "scores": scores.tolist(),
    }


def sweep_plot(run_dir: str | None = None):
    """Load latest run and plot mean_attn vs cosine with sign_match bars."""
    import json

    if run_dir is None:
        runs = sorted(Path(results_dir(__file__)).parent.glob("*"))
        run_dir = runs[-1] if runs else None

    if run_dir is None:
        return None, "No runs found"

    payload_path = Path(run_dir) / "benchmark.json"
    if not payload_path.exists():
        return None, f"No benchmark.json in {run_dir}"

    with open(payload_path) as f:
        payload = json.load(f)

    sweep = payload["sweep"]
    baseline = payload["linear_baseline_sweep"]

    cosines = [r["cosine"] for r in sweep]
    mean_attn = [r["mean_attn"] for r in sweep]
    sign_match = [r["sign_match"] for r in sweep]
    base_mean = [r["mean_attn"] for r in baseline]
    base_match = [r["sign_match"] for r in baseline]

    import plotly.graph_objects as go

    fig = go.Figure()

    # Mean attention scores
    fig.add_trace(go.Scatter(
        x=cosines, y=mean_attn, mode="lines+markers",
        name="Sharpened Attention (mean)", line=dict(color="blue", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=cosines, y=base_mean, mode="lines+markers",
        name="Linear Baseline (mean)", line=dict(color="gray", width=2, dash="dash")
    ))

    # Sign match as bar overlay (secondary y-axis)
    fig.add_trace(go.Bar(
        x=cosines, y=sign_match, name="Sign Match (ours)",
        opacity=0.3, marker_color="blue", yaxis="y2", width=0.12
    ))
    fig.add_trace(go.Bar(
        x=cosines, y=base_match, name="Sign Match (baseline)",
        opacity=0.3, marker_color="gray", yaxis="y2", width=0.12
    ))

    # Threshold line
    fig.add_vline(x=0, line_dash="dot", line_color="red", annotation_text="τ=0")

    fig.update_layout(
        title="Sign Threshold: Mean Pre-Softmax Score & Sign Accuracy vs Cosine",
        xaxis_title="cos(q, k)",
        yaxis_title="Mean Pre-Softmax Score",
        yaxis2=dict(title="Sign Match Fraction", overlaying="y", side="right", range=[0, 1.05]),
        legend=dict(x=0.02, y=0.98),
        height=500,
        template="plotly_white",
    )

    return fig, f"Loaded run: {run_dir.name}"


# Available run directories for dropdown
def get_run_dirs():
    base = Path(results_dir(__file__)).parent
    runs = sorted(base.glob("*"), reverse=True)
    return [str(r) for r in runs] if runs else ["(no runs yet)"]


with gr.Blocks() as demo:
    gr.Markdown("# Attention Sign Threshold — First Pass Demo")

    with gr.Tab("Demo"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Single Cosine Slice Explorer")
                cosine_slider = gr.Slider(-1.0, 1.0, value=0.2, step=0.1, label="cos(q, k)")
                n_pairs = gr.Slider(100, 5000, value=500, step=100, label="# pairs")
                beta_slider = gr.Slider(1.0, 100.0, value=20.0, step=1.0, label="β (sharpness)")
                seed_input = gr.Number(value=42, label="Seed", precision=0)
                run_btn = gr.Button("Generate & Score", variant="primary")

                with gr.Row():
                    mean_out = gr.Number(label="Mean Score")
                    std_out = gr.Number(label="Std Score")
                    match_out = gr.Number(label="Sign Match Fraction")

            with gr.Column(scale=1):
                hist_plot = gr.Plot(label="Score Distribution")

        def on_run(cos, n, beta, seed):
            result = run_demo(float(cos), int(n), float(beta), int(seed))
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Bar(x=result["hist_bins"][:-1], y=result["hist_counts"],
                                 width=np.diff(result["hist_bins"])[0], opacity=0.7))
            fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero")
            fig.update_layout(
                title=f"Scores at cos={cos:.1f} (target sign: {result['target_sign']:+d})",
                xaxis_title="Pre-softmax Score", yaxis_title="Count",
                height=300, template="plotly_white"
            )
            return result["mean_score"], result["std_score"], result["sign_match"], fig

        run_btn.click(
            on_run,
            inputs=[cosine_slider, n_pairs, beta_slider, seed_input],
            outputs=[mean_out, std_out, match_out, hist_plot],
        )

        gr.Markdown("---")
        gr.Markdown("### Canonical Sweep Results (latest run)")
        run_dropdown = gr.Dropdown(choices=get_run_dirs(), label="Select Run", value=get_run_dirs()[0])
        sweep_fig = gr.Plot()
        run_status = gr.Markdown()

        def on_sweep_change(run_dir):
            fig, msg = sweep_plot(run_dir)
            return fig, msg

        run_dropdown.change(on_sweep_change, inputs=run_dropdown, outputs=[sweep_fig, run_status])
        demo.load(on_sweep_change, inputs=run_dropdown, outputs=[sweep_fig, run_status])

    with gr.Tab("Benchmark"):
        goal_dir = Path(__file__).parent.parent
        benchmark_panel(goal_dir)

if __name__ == "__main__":
    demo.launch()