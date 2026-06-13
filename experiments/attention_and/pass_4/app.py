"""Gradio app: Demo (mechanism viz) + Benchmark (cross-attempt history)."""
import numpy as np
import gradio as gr

from agentic.experiments import benchmark_panel

GOAL_DIR = __file__  # benchmark_panel resolves the goal dir from this file's location

D = 64
N_POS = 100
RHO = 0.3
BETA = 2.0
COND = [(0, 0), (1, 0), (0, 1), (1, 1)]
COND_LBL = ["neither", "A only", "B only", "A AND B"]


def _features(cos_val, seed):
    rng = np.random.default_rng(int(seed))
    q_A = rng.normal(size=D); q_A /= np.linalg.norm(q_A)
    o = rng.normal(size=D); o -= o @ q_A * q_A
    n = np.linalg.norm(o)
    o = o / n if n > 1e-9 else o
    q_B = cos_val * q_A + np.sqrt(max(0.0, 1 - cos_val ** 2)) * o
    q_B /= np.linalg.norm(q_B)
    res = rng.normal(size=(N_POS, D)) * 0.5
    fa = rng.random(N_POS) < RHO
    fb = rng.random(N_POS) < RHO
    res[fa] += q_A * 2.0
    res[fb] += q_B * 2.0
    return q_A, q_B, res, fa, fb


def _scores(q_A, q_B, res, kind):
    pA, pB = res @ q_A, res @ q_B
    if kind == "AND (product)":
        return BETA * np.maximum(pA, 0) * np.maximum(pB, 0)
    return pA + pB  # linear baseline


def demo_fig(cos_val, kind, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_A, q_B, res, fa, fb = _features(cos_val, seed)
    sc = _scores(q_A, q_B, res, kind)
    attn = np.exp(sc - sc.max()); attn /= attn.sum()

    means = []
    for a, b in COND:
        m = (fa == bool(a)) & (fb == bool(b))
        means.append(float(attn[m].mean()) if m.any() else 0.0)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    colors = ["#bbbbbb", "#bbbbbb", "#bbbbbb", "#d62728"]
    ax.bar(COND_LBL, np.array(means) * 100, color=colors)
    ax.axhline(100.0 / N_POS, ls="--", c="k", lw=1, label="uniform (1/N)")
    ax.set_ylabel("mean attention  (% of mass)")
    ax.set_title(f"{kind} — cos(q_A,q_B)={cos_val:.1f}")
    ax.legend()
    fig.tight_layout()
    return fig


with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_and — ReLU-gated bilinear AND head\n"
        "Mean attention mass per feature condition. The AND head should spike "
        "**only** at `A AND B`; the linear baseline also lights up single "
        "features. Push the cosine toward 1.0 to test superposition robustness."
    )
    with gr.Tabs():
        with gr.TabItem("Demo"):
            with gr.Row():
                cos = gr.Slider(0.0, 1.0, value=0.0, step=0.2, label="cos(q_A, q_B)")
                kind = gr.Radio(["AND (product)", "linear baseline"],
                                value="AND (product)", label="head")
            plot = gr.Plot()
            demo.load(demo_fig, inputs=[cos, kind], outputs=plot)
            cos.change(demo_fig, inputs=[cos, kind], outputs=plot)
            kind.change(demo_fig, inputs=[cos, kind], outputs=plot)
        with gr.TabItem("Benchmark"):
            benchmark_panel(GOAL_DIR)

if __name__ == "__main__":
    demo.launch()
