import gradio as gr
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
from agentic.experiments import benchmark_panel, load_task, results_dir

task = load_task(__file__)

DEVICE = "cuda"
SHARPNESS = 50.0


def compute_attention(queries_np: np.ndarray, keys_np: np.ndarray) -> np.ndarray:
    """Compute attention weights using the hand-built sign threshold circuit."""
    q = torch.as_tensor(queries_np, dtype=torch.float32, device=DEVICE)
    k = torch.as_tensor(keys_np, dtype=torch.float32, device=DEVICE)
    dot = torch.einsum("bd,bd->b", q, k)
    with torch.no_grad():
        attn = torch.sigmoid(SHARPNESS * dot)
    return attn.detach().cpu().numpy().astype(np.float32)


def generate_pairs(cosine: float, n_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate query/key pairs with exact cosine similarity."""
    rng = np.random.default_rng(seed)
    d_model = 64

    q = rng.normal(size=(n_pairs, d_model)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-8

    sin_val = np.sqrt(max(0.0, 1.0 - cosine * cosine))
    ortho = rng.normal(size=(n_pairs, d_model)).astype(np.float32)
    ortho -= np.sum(ortho * q, axis=1, keepdims=True) * q
    ortho /= np.linalg.norm(ortho, axis=1, keepdims=True) + 1e-8

    k = cosine * q + sin_val * ortho
    k /= np.linalg.norm(k, axis=1, keepdims=True) + 1e-8

    return q, k


def run_demo(cosine: float, n_pairs: int, seed: int):
    """Generate pairs at given cosine and compute attention scores."""
    q, k = generate_pairs(cosine, n_pairs, seed)
    scores = compute_attention(q, k)

    target_sign = 1.0 if cosine > 0 else -1.0 if cosine < 0 else 0.0
    sign_match = float(np.mean(np.sign(scores - 0.5) == target_sign)) if target_sign != 0.0 else 0.0

    hist, bins = np.histogram(scores, bins=30, range=(0.0, 1.0))
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


def build_histogram_figure(result: dict) -> matplotlib.figure.Figure:
    """Build a matplotlib histogram figure for the demo."""
    fig, ax = plt.subplots(figsize=(6, 3))
    bins = np.array(result["hist_bins"])
    counts = np.array(result["hist_counts"])
    widths = np.diff(bins)
    centers = bins[:-1] + widths / 2

    ax.bar(centers, counts, width=widths * 0.9, color="#1f77b4", alpha=0.8, edgecolor="white")
    ax.axvline(x=0.5, color="red", linestyle=":", linewidth=2, label="Decision boundary (0.5)")
    ax.axvline(x=0.0, color="orange", linestyle=":", linewidth=1, alpha=0.5)
    ax.axvline(x=1.0, color="orange", linestyle=":", linewidth=1, alpha=0.5)

    cos = result["cosine"]
    target = result["target_sign"]
    ax.set_title(f"Attention Scores at cos={cos:.1f} (Target sign: {target:+.0f})", fontsize=11)
    ax.set_xlabel("Attention Weight", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def load_sweep_from_run(run_dir: Path):
    """Load sweep data from a benchmark.json file."""
    payload_path = run_dir / "benchmark.json"
    if not payload_path.exists():
        return None, f"No benchmark.json in {run_dir}"
    with open(payload_path) as f:
        payload = json.load(f)
    return payload, f"Loaded sweep from run: {run_dir.name}"


def build_sweep_figure(payload: dict) -> matplotlib.figure.Figure:
    """Build the canonical sweep plot with mean attention vs cosine."""
    sweep = payload["sweep"]
    cosines = [rec["cosine"] for rec in sweep]
    means = [rec["mean_attention"] for rec in sweep]
    stds = [rec["std_attention"] for rec in sweep]

    # Linear baseline: attention = max(0, cos)
    baseline_means = [max(0.0, c) for c in cosines]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Main curve: mean attention
    ax1.plot(cosines, means, "o-", color="#1f77b4", linewidth=2.5, markersize=6,
             label="Hand-built Sign Threshold (sharpness=50)")
    ax1.plot(cosines, baseline_means, "--", color="#8c8da0", linewidth=2,
             label="Linear Baseline (max(0, cos))")

    # Error bars
    ax1.errorbar(cosines, means, yerr=stds, fmt="none", color="#1f77b4",
                 alpha=0.5, capsize=3, capthick=1)

    ax1.axvline(x=0, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label="τ = 0")
    ax1.axhline(y=0.5, color="red", linestyle=":", linewidth=1.5, alpha=0.7)

    ax1.set_xlabel("cos(q, k)", fontsize=12)
    ax1.set_ylabel("Mean Attention Weight", fontsize=12, color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(-1.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # Twin axis for sign-match bars
    ax2 = ax1.twinx()
    sign_targets = np.sign(cosines)
    sign_matches = np.where(sign_targets == 0, 0.0,
                            (np.sign(np.array(means) - 0.5) == sign_targets).astype(float))
    baseline_signs = np.where(sign_targets == 0, 0.0,
                              (np.sign(np.array(baseline_means) - 0.5) == sign_targets).astype(float))

    width = 0.08
    x_pos = np.array(cosines)
    ax2.bar(x_pos - width/2, sign_matches, width=width, color="#1f77b4", alpha=0.4,
            label="Sign Match (Ours)", edgecolor="#1f77b4")
    ax2.bar(x_pos + width/2, baseline_signs, width=width, color="#8c8da0", alpha=0.2,
            label="Sign Match (Baseline)", edgecolor="#8c8da0")

    ax2.set_ylabel("Sign Match Fraction", fontsize=12, color="#333333")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="#333333")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=10)

    ax1.set_title("Canonical Sweep: Attention Weight vs Cosine Similarity", fontsize=13, pad=15)
    fig.tight_layout()
    return fig


def get_run_dirs():
    """Get list of available run directories."""
    base = Path(results_dir(__file__)).parent
    runs = sorted(base.glob("*"), reverse=True)
    return [str(r) for r in runs] if runs else ["(no runs yet)"]


with gr.Blocks() as demo:
    gr.Markdown("# Attention Sign Threshold — Hand-Built Sign Threshold Circuit")

    with gr.Tab("Demo"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Single Cosine Slice Explorer")
                cos_slider = gr.Slider(-1.0, 1.0, value=0.0, step=0.1, label="cos(q, k)")
                n_pairs = gr.Slider(100, 5000, value=500, step=100, label="# pairs")
                seed_input = gr.Number(value=42, label="Seed", precision=0)
                run_btn = gr.Button("Generate & Score", variant="primary")

                with gr.Row():
                    mean_out = gr.Number(label="Mean Attention", precision=4)
                    std_out = gr.Number(label="Std", precision=4)
                    match_out = gr.Number(label="Sign Match Fraction", precision=4)

            with gr.Column(scale=1):
                hist_plot = gr.Plot()

        def demo_compute(cos, n, seed):
            result = run_demo(float(cos), int(n), int(seed))
            fig = build_histogram_figure(result)
            return result["mean_score"], result["std_score"], result["sign_match"], fig

        run_btn.click(
            demo_compute,
            inputs=[cos_slider, n_pairs, seed_input],
            outputs=[mean_out, std_out, match_out, hist_plot]
        )

        gr.Markdown("---")
        gr.Markdown("### Canonical Sweep (loaded from selected run)")
        with gr.Row():
            run_dropdown = gr.Dropdown(choices=get_run_dirs(), label="Select Run",
                                       value=get_run_dirs()[0] if get_run_dirs() else "(no runs yet)")
            sweep_fig = gr.Plot()

        def on_dropdown(run_dir_str):
            if run_dir_str == "(no runs yet)" or not run_dir_str:
                return None, "No runs available"
            run_dir = Path(run_dir_str)
            payload, msg = load_sweep_from_run(run_dir)
            if payload is None:
                return None, msg
            fig = build_sweep_figure(payload)
            return fig, msg

        run_dropdown.change(on_dropdown, inputs=run_dropdown, outputs=[sweep_fig, gr.Textbox(visible=False)])
        demo.load(on_dropdown, inputs=run_dropdown, outputs=[sweep_fig, gr.Textbox(visible=False)])

    with gr.Tab("Benchmark"):
        goal_dir = Path(__file__).parent.parent
        benchmark_panel(goal_dir)


if __name__ == "__main__":
    demo.launch()