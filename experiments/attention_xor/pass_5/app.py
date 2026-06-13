"""Gradio app for attention_xor / pass_5.

Demo tab: the hand-built attention+ReLU XOR circuit, shown as
  (1) an interactive single-example walk-through exposing s = A+B, the CLS
      attention weights, and the bump logit, and
  (2) the full truth table + a bar chart contrasting the real circuit (2 ReLU)
      against the ablated one (1 ReLU) that collapses to the linear floor.

All torch/CUDA work is deferred to event handlers (demo.load / .change) so the
module imports cleanly during the boot-check on a GPU-less host.
"""

import os
import sys

import pandas as pd
import gradio as gr
import torch

from agentic.experiments import benchmark_panel

# Make this attempt's directory importable regardless of the boot-check's cwd
# (the boot-check imports app.py from elsewhere, so `import main` would otherwise
# fail with ModuleNotFoundError).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import main  # noqa: E402
from main import explain, truth_table_records  # noqa: E402

# Display-only safety net: the scored path in main.py always runs on CUDA, but
# the structural boot-check may import this module on a host without a visible
# GPU. Fall back to CPU *for the demo rendering only* so the app still imports
# and its handlers run. main.py's benchmark path is untouched.
if not torch.cuda.is_available():
    main.DEVICE = "cpu"

GOAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TABLE_COLS = [
    "cell", "s (=A+B)", "attn_A", "attn_B", "xor", "logit", "pred", "ablation_pred"
]
EMPTY_TABLE = pd.DataFrame(columns=TABLE_COLS)
EMPTY_BAR = pd.DataFrame(columns=["cell", "variant", "logit"])


def single_view(a_val: bool, b_val: bool) -> str:
    e = explain(int(a_val), int(b_val))
    w = e["attn_cls"]
    ok = "✅ correct" if e["pred"] == e["xor"] else "❌ wrong"
    return (
        f"### Input  A={e['A']}, B={e['B']}  →  XOR = **{e['xor']}**\n\n"
        f"**Step 1 — attention pools the bits.** CLS attention weights over "
        f"`[CLS, A_tok, B_tok, SEP]` = "
        f"`[{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}, {w[3]:.3f}]` "
        f"(≈ 0.5 on each feature token). Pooled sum **s = A + B = {e['s']:.3f}**.\n\n"
        f"**Step 2 — ReLU bump.** "
        f"`logit = 0.5 − relu(s−1) − relu(1−s)` = **{e['logit']:.3f}**  "
        f"→ prediction **XOR={e['pred']}**  ({ok}).\n\n"
        f"**Ablation (drop one ReLU).** `logit = 0.5 − relu(s−1)` = "
        f"{e['ablation_logit']:.3f} → predicts XOR={e['ablation_pred']} "
        f"(this monotone variant is linearly separable and cannot do XOR)."
    )


def table_df() -> pd.DataFrame:
    return pd.DataFrame(truth_table_records())[TABLE_COLS]


def bar_df() -> pd.DataFrame:
    rows = []
    for r in truth_table_records():
        rows.append({"cell": r["cell"], "variant": "circuit (2 ReLU)", "logit": r["logit"]})
        rows.append({"cell": r["cell"], "variant": "ablation (1 ReLU)", "logit": r["ablation_logit"]})
    return pd.DataFrame(rows)


def refresh():
    return single_view(False, False), table_df(), bar_df()


with gr.Blocks() as demo:
    gr.Markdown("# attention_xor — pass_5")
    gr.Markdown(
        "**Hand-built single attention head + 2-unit ReLU MLP.** Attention pools "
        "the two bits into their sum `s = A+B`; a ReLU *bump* "
        "`logit = 0.5 − relu(s−1) − relu(1−s)` fires only at `s == 1`, i.e. exactly "
        "XOR. No training, no learned weights — all compute on CUDA."
    )

    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                A_toggle = gr.Checkbox(label="A = 1", value=False)
                B_toggle = gr.Checkbox(label="B = 1", value=False)
            single_md = gr.Markdown()

            gr.Markdown("### Full truth table (all four input cells)")
            table = gr.Dataframe(value=EMPTY_TABLE, interactive=False)

            gr.Markdown(
                "### Logit per cell: circuit vs. ablation\n"
                "The **circuit** logit is positive only on the XOR=1 cells "
                "(`A=0,B=1` and `A=1,B=0`). The **ablation** (one ReLU removed) is "
                "positive on three cells — a linearly-separable NAND that the linear "
                "floor already matches, so it captures none of the XOR headroom."
            )
            bar = gr.BarPlot(
                value=EMPTY_BAR,
                x="cell",
                y="logit",
                color="variant",
                title="logit by input cell (positive ⇒ predict XOR=1)",
                y_lim=[-1.1, 1.1],
                height=320,
            )

            def on_change(a, b):
                return single_view(a, b)

            A_toggle.change(on_change, inputs=[A_toggle, B_toggle], outputs=single_md)
            B_toggle.change(on_change, inputs=[A_toggle, B_toggle], outputs=single_md)

            demo.load(refresh, inputs=[], outputs=[single_md, table, bar])

        with gr.Tab("Benchmark"):
            benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
