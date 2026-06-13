import gradio as gr
import numpy as np
from agentic.experiments import load_task, benchmark_panel

# Fixed random seed for demo consistency.
DEMO_SEED = 123

def _run_demo(cos_val, seed=DEMO_SEED):
    """Generate a demo sweep for a single cos_AB value using the static model."""
    d = 64
    nPos = 100
    rho = 0.3
    rng = np.random.default_rng(int(seed))

    # Build feature vectors with controlled cosine.
    cos = cos_val
    q_A = rng.normal(size=d)
    q_A = q_A / np.linalg.norm(q_A)

    ortho = rng.normal(size=d)
    ortho -= np.dot(ortho, q_A) * q_A
    ortho_norm = np.linalg.norm(ortho)
    if ortho_norm < 1e-10:
        ortho = np.zeros(d)
    else:
        ortho = ortho / ortho_norm

    q_B = cos * q_A + np.sqrt(max(0.0, 1.0 - cos**2)) * ortho
    q_B = q_B / np.linalg.norm(q_B)

    # Build the hand-built query projector (orthogonal for A and B).
    # Use a 2D slice of an orthogonal basis.
    mat = rng.normal(d, d)
    mat = mat @ np.diag([1.0, 0.98] + [-0.01] * (d-2))
    _, q = np.linalg.qr(mat)
    signs = np.sign(q[0, :])
    q = q * signs
    w_q = q.T[:, :2]   # (d, 2)

    # Residual: noisy + features A and B.
    res = rng.normal(size=(nPos, d)) * 0.5
    feat_A = (rng.random(nPos) < rho).astype(bool)
    feat_B = (rng.random(nPos) < rho).astype(bool)

    res[feat_A] += q_A * 2.0
    res[feat_B] += q_B * 2.0
    label = feat_A & feat_B

    # Probe projections.
    proj_A = np.dot(res, w_q[:, 0])
    proj_B = np.dot(res, w_q[:, 1])

    # AND-like conjunction.
    score = (proj_A * proj_B) / np.sqrt(d)

    # Compute per-condition means.
    labels = ["(0,0)", "(1,0)", "(0,1)", "(1,1)"]
    means = np.zeros(4)
    for i, (a, b) in enumerate([(0,0), (1,0), (0,1), (1,1)]):
        mask = (feat_A == (a != 0)) & (feat_B == (b != 0))
        means[i] = score[mask].mean()

    return means

def build_plot(cos_AB_val):
    """Return bar-chart data for mean scores per condition at the given cosine."""
    scores = _run_demo(cos_AB_val)
    return scores

with gr.Blocks(theme=gr.Themes.default()) as demo:
    gr.Markdown("# Attention AND (hand-built AND probe)")
    with gr.Tabs():
        with gr.TabItem("Demo"):
            gr.Markdown(
                "Bar chart of mean attention scores for the four presence conditions `α_A, α_B`:"
            )
            cos_slider = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.1, label="cos(e_A, e_B)",
                value=0.0, interactive=True
            )
            with gr.Row():
                chart = gr.BarPlot(
                    title="Mean attention score per condition",
                    y_label="Score",
                    x_title="Condition (α_A, α_B)",
                    width=600,
                    height=300,
                    tooltip=["Score"],
                )
            demo.load(fn=build_plot, inputs=cos_slider, outputs=chart)
            cos_slider.change(
                fn=build_plot,
                inputs=cos_slider,
                outputs=chart,
            )
        with gr.TabItem("Benchmark"):
            benchmark_panel(__file__)

if __name__ == "__main__":
    demo.launch()