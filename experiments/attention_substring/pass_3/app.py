"""Gradio app for attention_substring / pass_3.

Demo tab: shows that the hand-built induction circuit's best head points the
target position back at the source position — with (1) a headline bar comparing
the induction circuit vs the prev-token-head ablation vs the random baseline,
(2) a per-cell detection table across pattern length x distance, and (3) the
raw layer-1 attention from target_pos for picked example sequences, where the
single tall bar should sit exactly on source_pos.
"""

import json
from pathlib import Path

import gradio as gr
import pandas as pd

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ATTEMPT_DIR / "results"
GOAL_DIR = ATTEMPT_DIR.parent


def list_runs():
    if not RESULTS_DIR.exists():
        return []
    runs = sorted([p.name for p in RESULTS_DIR.iterdir() if p.is_dir()], reverse=True)
    return runs


def _load_json(run, name):
    if not run:
        return None
    f = RESULTS_DIR / run / name
    if not f.exists():
        return None
    with open(f) as fh:
        return json.load(fh)


def comparison_df(run):
    comp = _load_json(run, "comparison.json")
    if comp is None:
        return pd.DataFrame({"model": [], "detection": []})
    return pd.DataFrame({
        "model": ["induction circuit", "prev-head ablated", "random baseline"],
        "detection": [comp["induction_detection"], comp["ablation_detection"], comp["random_baseline"]],
    })


def cells_df(run):
    comp = _load_json(run, "comparison.json")
    if comp is None:
        return pd.DataFrame()
    cells = comp["cells"]
    rows = []
    for L in (2, 3, 4):
        rows.append({
            "pattern_length": L,
            "dist=8": cells.get(f"plen{L}_dist8", 0.0),
            "dist=16": cells.get(f"plen{L}_dist16", 0.0),
            "dist=32": cells.get(f"plen{L}_dist32", 0.0),
        })
    return pd.DataFrame(rows)


def example_choices(run):
    ex = _load_json(run, "examples.json") or []
    return [e["label"] for e in ex]


def example_df(run, label):
    ex = _load_json(run, "examples.json") or []
    chosen = None
    for e in ex:
        if e["label"] == label:
            chosen = e
            break
    if chosen is None and ex:
        chosen = ex[0]
    if chosen is None:
        return pd.DataFrame({"position": [], "attn": [], "role": []})
    spos, tpos = chosen["source_pos"], chosen["target_pos"]
    rows = []
    for k, a in enumerate(chosen["attn"]):
        role = "source_pos" if k == spos else ("target_pos" if k == tpos else "other")
        rows.append({"position": k, "attn": a, "role": role})
    return pd.DataFrame(rows)


def refresh(run):
    return (
        comparison_df(run),
        cells_df(run),
        gr.Dropdown(choices=example_choices(run), value=(example_choices(run)[0] if example_choices(run) else None)),
        example_df(run, example_choices(run)[0] if example_choices(run) else None),
    )


with gr.Blocks(title="attention_substring / pass_3") as demo:
    gr.Markdown(
        "# Attention Substring — hand-built induction circuit (pass_3)\n"
        "A 2-layer, single-head, hand-set attention circuit. **Layer 0** is a "
        "previous-token head; **Layer 1** matches `token[q-1]` against `token[k]` "
        "with an earliest-wins bias, so the target position attends to the *first* "
        "occurrence's last token (`source_pos`)."
    )

    with gr.Tabs():
        with gr.TabItem("Demo"):
            run_dd = gr.Dropdown(label="Run", choices=list_runs(),
                                 value=(list_runs()[0] if list_runs() else None))

            gr.Markdown("### Detection: circuit vs ablation vs chance\n"
                        "`correct_top1` = best head's argmax attention from `target_pos` equals "
                        "`source_pos`, averaged over all 450 sequences. Ablating the prev-token "
                        "head (Layer 0) is the causal-faithfulness check — it should collapse to ~0.")
            comp_plot = gr.BarPlot(value=comparison_df(list_runs()[0] if list_runs() else None),
                                   x="model", y="detection", y_lim=[0, 1],
                                   title="Substring-detection rate", height=300)

            gr.Markdown("### Operating range: detection by pattern length x distance")
            cells_tbl = gr.Dataframe(value=cells_df(list_runs()[0] if list_runs() else None),
                                     label="mean correct_top1 per cell", interactive=False)

            gr.Markdown("### Example: Layer-1 attention from `target_pos`\n"
                        "One tall bar should land on `source_pos` (orange). This is the raw "
                        "attention distribution the benchmark reads — not a hand-injected answer.")
            ex_dd = gr.Dropdown(label="Example sequence",
                                choices=example_choices(list_runs()[0] if list_runs() else None),
                                value=(example_choices(list_runs()[0] if list_runs() else None)[0]
                                       if (list_runs() and example_choices(list_runs()[0])) else None))
            ex_plot = gr.BarPlot(
                value=example_df(list_runs()[0] if list_runs() else None,
                                 example_choices(list_runs()[0] if list_runs() else None)[0]
                                 if (list_runs() and example_choices(list_runs()[0])) else None),
                x="position", y="attn", color="role", y_lim=[0, 1],
                title="Attention weight from target_pos over key positions", height=320)

            run_dd.change(refresh, inputs=run_dd, outputs=[comp_plot, cells_tbl, ex_dd, ex_plot])
            ex_dd.change(example_df, inputs=[run_dd, ex_dd], outputs=ex_plot)

        with gr.TabItem("Benchmark"):
            benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
