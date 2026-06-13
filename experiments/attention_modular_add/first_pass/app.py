"""Gradio app for the first_pass Fourier/circular modular addition attempt.

Two tabs:
- Demo: interactive visualisation of the Fourier mechanism and predictions
- Benchmark: leaderboard across all attempts (injected via benchmark_panel)
"""

import gradio as gr
import numpy as np
import json
from pathlib import Path

from agentic.experiments import benchmark_panel, results_dir

GOAL_DIR = Path(__file__).parent.parent


def fourier_model_fn(a: np.ndarray, b: np.ndarray, modulus: int) -> np.ndarray:
    """Same Fourier mechanism as main.py for interactive demo."""
    n = len(a)
    K = min(modulus // 2, 32)
    freqs = np.arange(1, K + 1, dtype=np.float32)

    angles_a = 2 * np.pi * freqs[None, :] * a[:, None] / modulus
    angles_b = 2 * np.pi * freqs[None, :] * b[:, None] / modulus

    emb_a = np.exp(1j * angles_a)
    emb_b = np.exp(1j * angles_b)
    emb_sum = emb_a * emb_b

    c_vals = np.arange(modulus, dtype=np.float32)
    angles_c = 2 * np.pi * freqs[:, None] * c_vals[None, :] / modulus
    emb_c = np.exp(-1j * angles_c)

    logits_complex = emb_sum @ emb_c
    return logits_complex.real.astype(np.float32)


def predict_sum(a_val: int, b_val: int, modulus: int):
    """Run model on single (a, b) pair and return formatted results."""
    a_arr = np.array([a_val], dtype=np.int32)
    b_arr = np.array([b_val], dtype=np.int32)
    logits = fourier_model_fn(a_arr, b_arr, modulus)[0]
    true_sum = (a_val + b_val) % modulus
    pred = int(logits.argmax())
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()

    # Top-5 predictions
    top5_idx = np.argsort(logits)[-5:][::-1]
    top5_str = ", ".join(f"{int(i)}: {probs[i]:.3f}" for i in top5_idx)

    correct = "✓" if pred == true_sum else "✗"
    return (
        f"True: {true_sum}, Predicted: {pred} {correct}\n"
        f"Top-5: {top5_str}"
    )


def visualize_embeddings(modulus: int):
    """Generate a visualization of the Fourier embeddings on the unit circle."""
    # Use first few frequencies for visualization
    K_vis = min(4, modulus // 2)
    freqs = np.arange(1, K_vis + 1)

    # Embed all numbers 0..p-1 for the first frequency
    x_vals = np.arange(modulus)
    angles = 2 * np.pi * freqs[0] * x_vals / modulus

    # Create a simple text-based unit circle visualization
    lines = [f"Fourier embeddings for modulus p={modulus} (freq k=1):"]
    lines.append("Number → angle (deg) → (cos, sin)")
    for x in x_vals:
        angle_deg = angles[x] * 180 / np.pi
        cos_val = np.cos(angles[x])
        sin_val = np.sin(angles[x])
        lines.append(f"  {x:3d} → {angle_deg:6.1f}° → ({cos_val:6.3f}, {sin_val:6.3f})")

    # Show how addition works: a + b corresponds to angle addition
    lines.append("\nExample: 5 + 7 mod 17")
    a_ex, b_ex = 5, 7
    angle_a = 2 * np.pi * 1 * a_ex / modulus
    angle_b = 2 * np.pi * 1 * b_ex / modulus
    angle_sum = (angle_a + angle_b) % (2 * np.pi)
    sum_pred = int(round(angle_sum * modulus / (2 * np.pi))) % modulus
    true_sum = (a_ex + b_ex) % modulus
    lines.append(f"  Angle(5) = {angle_a*180/np.pi:.1f}°")
    lines.append(f"  Angle(7) = {angle_b*180/np.pi:.1f}°")
    lines.append(f"  Sum angle = {angle_sum*180/np.pi:.1f}° → predicts {sum_pred} (true: {true_sum})")

    return "\n".join(lines)


def load_latest_run():
    """Find the most recent run directory and load its benchmark.json."""
    results_base = results_dir(__file__).parent  # parent of the would-be new run dir
    if not results_base.exists():
        return "No runs yet. Run main.py first."
    run_dirs = sorted([d for d in results_base.iterdir() if d.is_dir()])
    if not run_dirs:
        return "No runs yet. Run main.py first."
    latest = run_dirs[-1]
    bench_path = latest / "benchmark.json"
    if bench_path.exists():
        with open(bench_path) as f:
            return json.dumps(json.load(f), indent=2)
    return f"No benchmark.json in {latest}"


with gr.Blocks() as demo:
    gr.Markdown("# attention_modular_add — first_pass (Fourier/circular)")

    with gr.Tab("Demo"):
        gr.Markdown(
            "Interactive demo of the hand-built Fourier mechanism. "
            "Numbers are embedded as complex exponentials on the unit circle; "
            "addition corresponds to multiplication (angle addition)."
        )

        with gr.Row():
            with gr.Column():
                a_in = gr.Number(label="a", value=5, precision=0, minimum=0)
                b_in = gr.Number(label="b", value=7, precision=0, minimum=0)
                mod_in = gr.Dropdown(
                    label="Modulus p",
                    choices=[11, 13, 17, 37, 53, 113],
                    value=17,
                )
                btn = gr.Button("Predict (a + b) mod p", variant="primary")
                out_text = gr.Textbox(label="Result", lines=4)

            with gr.Column():
                gr.Markdown("### Embedding Visualization")
                vis_mod = gr.Dropdown(
                    label="Modulus for visualization",
                    choices=[11, 13, 17, 37, 53, 113],
                    value=17,
                )
                vis_btn = gr.Button("Show Embeddings")
                vis_text = gr.Textbox(label="Embeddings", lines=20, max_lines=30)

        btn.click(predict_sum, inputs=[a_in, b_in, mod_in], outputs=out_text)
        vis_btn.click(visualize_embeddings, inputs=vis_mod, outputs=vis_text)

        # Auto-run on load for immediate feedback
        demo.load(predict_sum, inputs=[a_in, b_in, mod_in], outputs=out_text)
        demo.load(visualize_embeddings, inputs=vis_mod, outputs=vis_text)

    with gr.Tab("Benchmark"):
        gr.Markdown("### Benchmark History (across all attempts)")
        benchmark_panel(str(GOAL_DIR))

    with gr.Tab("Latest Run Metrics"):
        gr.Markdown("Raw `benchmark.json` from the most recent run of this attempt.")
        refresh_btn = gr.Button("Refresh")
        metrics_text = gr.Textbox(label="benchmark.json", lines=30, max_lines=50)
        refresh_btn.click(load_latest_run, outputs=metrics_text)
        demo.load(load_latest_run, outputs=metrics_text)


if __name__ == "__main__":
    demo.launch()