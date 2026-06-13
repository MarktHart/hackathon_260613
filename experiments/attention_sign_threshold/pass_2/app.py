import gradio as gr
import numpy as np
import torch
import plotly.graph_objects as go
from pathlib import Path
from agentic.experiments import benchmark_panel, load_task, results_dir

task = load_task(__file__)


def run_demo(cosine: float, n_pairs: int, beta: float, seed: int):
    """Generate a random batch of query/key pairs at the given cosine and compute the trained head's scores."""
    rng = np.random.default_rng(seed)
    d_model = 64

    # Generate pairs with exact cosine similarity
    q = rng.normal(size=(n_pairs, d_model))
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12

    sin_val = np.sqrt(max(0.0, 1.0 - cosine * cosine))
    ortho = rng.normal(size=(n_pairs, d_model))
    ortho -= np.sum(ortho * q, axis=1, keepdims=True) * q
    ortho /= np.linalg.norm(ortho, axis=1, keepdims=True) + 1e-12

    k = cosine * q + sin_val * ortho
    k /= np.linalg.norm(k, axis=1, keepdims=True) + 1e-12

    # Load the trained head from disk; we pre-compute once per run
    from experiments.attention_sign_threshold.pass_2.main import head
    device = "cpu"
    score = head(torch.tensor(q, dtype=torch.float32).unsqueeze(1), torch.tensor(k, dtype=torch.float32).unsqueeze(1)).squeeze(-1).detach().cpu().numpy()

    target_sign = 1.0 if cosine > 0 else -1.0 if cosine < 0 else 0.0
    sign_match = float(np.mean(np.sign(score) == target_sign)) if target_sign != 0.0 else 0.0

    hist, bins = np.histogram(score, bins=30, range=(-0.1, 1.1))
    return {
        "cosine": cosine,
        "target_sign": target_sign,
        "mean_score": float(np.mean(score)),
        "std_score": float(np.std(score)),
        "sign_match": sign_match,
        "hist_counts": hist.tolist(),
        "hist_bins": bins.tolist(),
        "scores": score.tolist(),
    }


def sweep_plot(run_dir: str | None = None):
    """Load the latest run and plot mean_attention vs cosine with an overlay of sign-match bar chart."""
    if run_dir is None:
        runs = sorted(Path(results_dir(__file__)).parent.glob("*"), reverse=True)
        run_dir = runs[0] if runs else None

    if run_dir is None or not Path(run_dir).exists():
        return None, "No runs found under result directories"

    payload_path = Path(run_dir) / "benchmark.json"
    if not payload_path.exists():
        return None, f"No benchmark.json in {run_dir}"

    with open(payload_path) as f:
        payload = json.load(f)

    sweep = payload["sweep"]
    cosines = [rec["cosine"] for rec in sweep]
    means = [rec["mean_attention"] for rec in sweep]
    # Compute sign-match per bin: compare mean_attention to a binary target
    sign_targets = np.sign(cosines)
    sign_matches = np.where(sign_targets == 0, 0.0, (np.sign(means) == sign_targets).astype(float))

    # Linear ramp baseline for comparison
    baseline_means = [max(0.0, float(c)) for c in cosines]
    baseline_signs = [1.0 if c > 0 else 0.0 for c in cosines]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cosines, y=means, mode="lines+markers", name="Trained Sign Threshold Head", line=dict(width=3, color="#1f77b4")))
    fig.add_trace(go.Scatter(x=cosines, y=baseline_means, mode="lines", name="Linear Baseline (ramp)", line=dict(width=2, color="#8c8da0", dash="dot")))
    fig.add_trace(go.Bar(x=cosines, y=sign_matches, marker_color="#1f77b4", opacity=0.3, name="Sign Match (ours)"))
    fig.add_trace(go.Bar(x=cosines, y=baseline_signs, marker_color="#8c8da0", opacity=0.15, name="Sign Match (baseline)"))

    fig.add_vline(x=0, line_dash="dot", line_color="red", annotation_text="τ=0")

    fig.update_layout(
        title="Trained Attention Sign Threshold vs Cosine Similarity",
        xaxis_title="cos(q, k)",
        yaxis_title="Mean Attention Weight",
        yaxis2=dict(title="Sign Match Fraction", overlaying="y", side="right", range=[0, 1.05]),
        legend=dict(x=0.02, y=0.98),
        height=460,
        template="plotly_white",
    )

    return fig, f"Loaded sweep from run: {Path(run_dir).name}"


def get_run_dirs():
    base = Path(results_dir(__file__)).parent
    runs = sorted(base.glob("*"), reverse=True)
    return [str(r) for r in runs] if runs else ["(no runs yet)"]


with gr.Blocks() as demo:
    gr.Markdown("# Attention Sign Threshold — Trained Sign Threshold Head")

    with gr.Tab("Demo"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Single Cosine Slice Explorer")
                cos_slider = gr.Slider(-1.0, 1.0, value=0.2, step=0.1, label="cos(q, k)")
                n_pairs = gr.Slider(100, 5000, value=500, step=100, label="# pairs")
                beta_slider = gr.Slider(1.0, 30.0, value=15.0, step=1.0, label="Sharpness (β)")
                seed_input = gr.Number(value=42, label="Seed", precision=0)
                run_btn = gr.Button("Generate & Score", variant="primary")

                with gr.Row():
                    mean_out = gr.Number(label="Mean Score / Attention")
                    std_out = gr.Number(label="Std")
                    match_out = gr.Number(label="Sign Match Fraction")

            with gr.Column(scale=1):
                hist_plot = gr.Plot().style(width="100%", height="200px")

        # Compute demo results
        def demo_compute(cos, n, beta, seed):
            result = run_demo(float(cos), int(n), float(beta), int(seed))
            fig = go.Figure()
            fig.add_trace(go.Bar(x=result["hist_bins"][:-1], y=result["hist_counts"],
                                 marker_color="#1f77b4", opacity=0.8, width=np.diff(result["hist_bins"])[0]))
            fig.add_vline(x=0, line_dash="dot", line_color="red", annotation_text="Zero")
            fig.add_vline(x=1.0, line_dash="dot", line_color="orange", annotation_text="1.0")
            fig.add_vline(x=-1.0, line_dash="dot", line_color="orange", annotation_text="-1.0")
            fig.update_layout(
                title=f"Attention Scores at cos={cos:.1f} (Target: {result['target_sign']:+d})",
                xaxis_title="Pre-softmax Attention Score",
                yaxis_title="Count",
                height=260,
                template="plotly_white",
            )
            return result["mean_score"], result["std_score"], result["sign_match"], fig

        run_btn.click(
            demo_compute,
            inputs=[cos_slider, n_pairs, beta_slider, seed_input],
            outputs=[mean_out, std_out, match_out, hist_plot]
        )

        gr.Markdown("---")
        gr.Markdown("### Canonical Sweep (loaded from latest run)")
        with gr.Row():
            run_dropdown = gr.Dropdown(choices=get_run_dirs(), label="Select Run", value=get_run_dirs()[0] if get_run_dirs() else "(no runs yet)")
            sweep_fig = gr.Plot().style(width="100%", height="500px")

        def on_dropdown(run_dir):
            fig, msg = sweep_plot(run_dir)
            return fig, msg

        run_dropdown.change(on_dropdown, inputs=run_dropdown, outputs=[sweep_fig, gr.update()])
        demo.load(on_dropdown, inputs=run_dropdown, outputs=[sweep_fig, gr.update()])

    with gr.Tab("Benchmark"):
        goal_dir = Path(__file__).parent.parent
        benchmark_panel(goal_dir)


if __name__ == "__main__":
    demo.launch()