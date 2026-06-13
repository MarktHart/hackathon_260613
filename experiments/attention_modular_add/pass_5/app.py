"""Gradio app for pass_5 Fourier attention head.

Demo tab: pick (a,b), see the per-frequency q·k contribution cos(2πk(a+b)/p) as a
bar chart and the summed attention logit over every candidate a+b — the peak at
the true a+b makes the mechanism legible. Benchmark tab: cross-attempt panel.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import gradio as gr
from agentic.experiments import benchmark_panel

THIS_DIR = Path(__file__).parent
GOAL_DIR = THIS_DIR.parent

P = 97
N_FREQ = P // 2


def per_freq_contrib(a, b):
    a, b = int(a) % P, int(b) % P
    k = np.arange(1, N_FREQ + 1)
    vals = np.cos(2 * np.pi * k * (a + b) / P)
    return pd.DataFrame({"frequency": k, "q·k contribution": vals})


def attention_over_sums(a, b):
    a, b = int(a) % P, int(b) % P
    s = np.arange(P)
    k = np.arange(1, N_FREQ + 1)[:, None]
    logits = np.cos(2 * np.pi * k * ((a + b) - s) / P).sum(axis=0)
    return pd.DataFrame({"candidate a+b mod p": s, "attention logit": logits})


def summary(a, b):
    a, b = int(a) % P, int(b) % P
    return f"a={a}, b={b}  →  (a+b) mod {P} = **{(a + b) % P}**  (logit peaks here)"


with gr.Blocks() as demo:
    gr.Markdown("# attention_modular_add / pass_5 — Hand-built Fourier head")
    gr.Markdown(
        "Query(a)=`[sin,cos](2πk a/p)`, Key(b)=conjugate `[-sin,cos](2πk b/p)` on the "
        "same channels ⇒ `q·k = Σₖ cos(2πk(a+b)/p)`, which peaks at `a+b (mod p)`."
    )

    with gr.Tab("Demo"):
        with gr.Row():
            a_in = gr.Slider(0, P - 1, value=12, step=1, label="a")
            b_in = gr.Slider(0, P - 1, value=23, step=1, label="b")
        out_md = gr.Markdown()
        gr.Markdown("**Per-frequency q·k contribution** — all 48 frequencies cohere at the answer.")
        bar = gr.BarPlot(x="frequency", y="q·k contribution", height=240)
        gr.Markdown("**Summed attention logit over every candidate sum** — sharp peak at the true a+b.")
        line = gr.LinePlot(x="candidate a+b mod p", y="attention logit", height=240)

        def update(a, b):
            return summary(a, b), per_freq_contrib(a, b), attention_over_sums(a, b)

        a_in.change(update, [a_in, b_in], [out_md, bar, line])
        b_in.change(update, [a_in, b_in], [out_md, bar, line])
        demo.load(update, [a_in, b_in], [out_md, bar, line])

    with gr.Tab("Benchmark"):
        benchmark_panel(str(GOAL_DIR))


if __name__ == "__main__":
    demo.launch()
