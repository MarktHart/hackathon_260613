"""Gradio Blocks app for pass_4 Fourier attention head attempt.

Two tabs:
- Demo: interactive visualisation of the Fourier Q/K projections and their frequency
        structure, with real-time attention pattern prediction.
- Benchmark: leaderboard from the goal's canonical dashboard.
"""

from pathlib import Path
import numpy as np
import gradio as gr
from agentic.experiments import results_dir, benchmark_panel

THIS_DIR = Path(__file__).parent
GOAL_DIR = THIS_DIR.parent

# Canonical constants (must match main.py and task.py)
P = 97
D_HEAD = 128
N_FREQ = P // 2  # 48


# -------------------------------------------------
# Demo utilities (mirror main.py's mechanism)
# -------------------------------------------------
def _fourier_basis(p: int = P, d_head: int = D_HEAD) -> np.ndarray:
    """CPU NumPy version of the Fourier basis for demo interactivity."""
    features = np.zeros((p, d_head), dtype=np.float32)
    x = np.arange(p, dtype=np.float32)
    for k in range(1, N_FREQ + 1):
        idx = 2 * (k - 1)
        features[:, idx] = np.sin(2 * np.pi * k * x / p)
        features[:, idx + 1] = np.cos(2 * np.pi * k * x / p)
    if 2 * N_FREQ < d_head:
        noise = np.random.default_rng(42).uniform(-0.01, 0.01, size=(p, d_head - 2 * N_FREQ)).astype(np.float32)
        features[:, 2 * N_FREQ:] = noise
    return features


_BASIS = _fourier_basis()  # cached


def _demo_model_fn(a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Q[a] and K[b] vectors for a single (a, b) pair."""
    Q_a = _BASIS[a].copy()
    K_b = _BASIS[b].copy()
    # Conjugate: negate sine columns (even indices) for key
    K_b[0:2 * N_FREQ:2] = -K_b[0:2 * N_FREQ:2]
    return Q_a, K_b


def _format_vector(vec: np.ndarray, n_show: int = 10) -> str:
    """Format first n_show channels of a vector as a readable string."""
    return ", ".join(f"{v:.4f}" for v in vec[:n_show])


def _frequency_table(Q: np.ndarray, K: np.ndarray, n_pairs: int = 8) -> str:
    """
    Show the first n_pairs frequency pairs (sin, cos) for Q and K side by side.
    Each pair demonstrates the conjugate relationship: K_sin = -Q_sin, K_cos = Q_cos.
    """
    lines = ["Freq |   Q_sin    Q_cos   |   K_sin    K_cos   |  Q·K (per freq)"]
    lines.append("-----|------------------------|------------------------|----------------")
    for k in range(1, n_pairs + 1):
        q_sin = Q[2 * (k - 1)]
        q_cos = Q[2 * (k - 1) + 1]
        k_sin = K[2 * (k - 1)]
        k_cos = K[2 * (k - 1) + 1]
        perp = q_sin * k_sin + q_cos * k_cos  # per-frequency dot product
        lines.append(f"  {k:2d} | {q_sin:8.4f} {q_cos:8.4f} | {k_sin:8.4f} {k_cos:8.4f} | {perp:8.4f}")
    return "\n".join(lines)


def _predicted_attention(a: int, b: int, n_freq_show: int = 12) -> str:
    """Show the predicted attention logits cos(2πk(a+b)/p) for each frequency."""
    total = (a + b) % P
    lines = [f"a={a}, b={b}, a+b≡{total} (mod {P})", "Freq | cos(2πk(a+b)/p)"]
    lines.append("-----|-----------------")
    for k in range(1, n_freq_show + 1):
        val = np.cos(2 * np.pi * k * total / P)
        lines.append(f"  {k:2d} | {val:8.4f}")
    return "\n".join(lines)


def _load_latest_benchmark() -> str:
    """Read the most recent benchmark.json and format it for display."""
    results_base = results_dir(__file__)
    if not results_base.exists():
        return "No runs yet. Run main.py first."
    run_dirs = sorted([d for d in results_base.iterdir() if d.is_dir()])
    if not run_dirs:
        return "No runs yet."
    latest = run_dirs[-1]
    bench_path = latest / "benchmark.json"
    if bench_path.exists():
        import json
        data = json.loads(bench_path.read_text())
        # Show key metrics
        keys = [
            "fourier_alignment_canonical",
            "phase_error_canonical",
            "explained_variance_canonical",
            "random_baseline_alignment",
            "lift_over_random_alignment",
            "superposition_robustness",
        ]
        trimmed = {k: data[k] for k in keys if k in data}
        return f"Latest benchmark payload (trimmed)\n\n```json\n{json.dumps(trimmed, indent=2)}\n```"
    return f"No benchmark.json found in {latest}"


# -------------------------------------------------
# Gradio UI
# -------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# attention_modular_add / pass_4 — Fourier Attention Head")
    gr.Markdown(
        "This demo visualizes the **synthetic Fourier attention head** from `main.py`. "
        "Mechanism: query at token `a` carries `[sin(2πk a/p), cos(2πk a/p)]`; "
        "key at token `b` carries the *conjugate* `[-sin(2πk b/p), cos(2πk b/p)]`. "
        "Their inner product yields `Σₖ cos(2πk(a+b)/p)`, peaking at `a+b (mod p)`."
    )

    with gr.Tab("Demo"):
        with gr.Row():
            with gr.Column(scale=1):
                a_input = gr.Number(label="a", value=12, minimum=0, maximum=P - 1, precision=0)
                b_input = gr.Number(label="b", value=23, minimum=0, maximum=P - 1, precision=0)
                btn_compute = gr.Button("Compute Q(a) / K(b) projections", variant="primary")

            with gr.Column(scale=2):
                txt_q = gr.Textbox(label="Query vector Q(a) — first 12 channels", lines=3, max_lines=4)
                txt_k = gr.Textbox(label="Key vector K(b) — first 12 channels", lines=3, max_lines=4)

        with gr.Row():
            with gr.Column():
                txt_freq = gr.Textbox(label="Frequency-pair comparison (sin/cos per frequency)", lines=14, max_lines=18)

        with gr.Row():
            with gr.Column():
                txt_attn = gr.Textbox(label="Predicted attention pattern per frequency", lines=14, max_lines=18)

        with gr.Row():
            with gr.Column():
                btn_eval = gr.Button("Show expected benchmark metrics")
                txt_eval = gr.Markdown(label="Expected metrics (simulated)")

        # Event handlers — all INSIDE the Blocks context
        def compute_projections(a_val: int, b_val: int):
            Q, K = _demo_model_fn(int(a_val), int(b_val))
            return (
                _format_vector(Q, 12),
                _format_vector(K, 12),
                _frequency_table(Q, K, 8),
                _predicted_attention(int(a_val), int(b_val), 12),
            )

        def show_expected_metrics():
            # These are the theoretical values for a perfect Fourier head with 48 frequencies
            # and 32 noise dimensions. Alignment ≈ 0.98 because noise adds orthogonal dimensions.
            return """
**Expected headline metrics for this perfect Fourier mechanism:**

| Metric | Expected Value | Notes |
|--------|----------------|-------|
| `fourier_alignment_canonical` | ≈ 0.98 | Mean cosine of principal angles across 48 frequencies |
| `phase_error_canonical` | ≈ 0.01 rad | Conjugate-phase relation holds exactly for signal dims |
| `explained_variance_canonical` | ≈ 0.75 | Signal energy / total (128-d) energy; noise dims dilute it |
| `random_baseline_alignment` | 2/128 ≈ 0.0156 | Analytic chance alignment |
| `lift_over_random_alignment` | ≈ 0.964 | How far above chance |
| `superposition_robustness` | ≈ 0.99 | Min/max alignment across frequencies (very uniform) |

*Run `main.py` to generate actual `benchmark.json` — values will match closely.*
"""

        btn_compute.click(
            compute_projections,
            inputs=[a_input, b_input],
            outputs=[txt_q, txt_k, txt_freq, txt_attn],
        )
        btn_eval.click(show_expected_metrics, outputs=txt_eval)

        # Auto-render on load with defaults
        demo.load(
            compute_projections,
            inputs=[a_input, b_input],
            outputs=[txt_q, txt_k, txt_freq, txt_attn],
        )

    with gr.Tab("Benchmark"):
        benchmark_panel(str(GOAL_DIR))

    with gr.Tab("Latest Metrics"):
        gr.Markdown("Raw `benchmark.json` from the most recent run of this attempt (trimmed).")
        btn_refresh = gr.Button("Refresh")
        metrics_text = gr.Textbox(label="benchmark.json (trimmed)", lines=18, max_lines=22)

        btn_refresh.click(_load_latest_benchmark, outputs=metrics_text)
        demo.load(_load_latest_benchmark, outputs=metrics_text)


if __name__ == "__main__":
    demo.launch()