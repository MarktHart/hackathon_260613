import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np

from agentic.experiments import benchmark_panel

sys.path.insert(0, str(Path(__file__).parent))
from main import model_fn, ablate_no_attention, ablate_linear_readout

GOAL_DIR = str(Path(__file__).resolve().parents[1])


def _encode(a: int, b: int) -> np.ndarray:
    return np.array([[0, a + 1, b + 3, 5]], dtype=np.int32)


def truth_table():
    rows = []
    for a in (0, 1):
        for b in (0, 1):
            tok = _encode(a, b)
            full = float(model_fn(tok)[0])
            noatt = float(ablate_no_attention(tok)[0])
            lin = float(ablate_linear_readout(tok)[0])
            xor = a ^ b
            ok = "✅" if (full > 0) == bool(xor) else "❌"
            rows.append(
                f"| {a} | {b} | {xor} | {full:+.3f} | {int(full>0)} {ok} "
                f"| {int(noatt>0)} | {int(lin>0)} |"
            )
    header = (
        "| A | B | XOR | full logit | full pred | no-attn pred | linear pred |\n"
        "|---|---|-----|-----------|-----------|--------------|-------------|\n"
    )
    return header + "\n".join(rows)


def latest_ablation_md():
    runs = sorted(Path(__file__).parent.glob("results/*/ablations.json"))
    if not runs:
        return "_No ablation artefact yet — run main.py._"
    data = json.loads(runs[-1].read_text())
    lines = ["| ablation | acc @p=0.5 | linear floor |", "|---|---|---|"]
    for name, sw in data.items():
        c = next(r for r in sw if abs(r["p"] - 0.5) < 1e-9)
        lines.append(f"| {name} | {c['accuracy']:.3f} | {c['baseline']:.3f} |")
    return "\n".join(lines)


with gr.Blocks() as demo:
    gr.Markdown("# attention_xor — pass_6 (hand-built single attention head)")
    gr.Markdown(
        "One self-attention head (no MLP) hand-wired on CUDA: CLS attends "
        "equally to A and B, pools signed value features `x,y ∈ {±1}`, and a "
        "**quadratic** readout `(x−y)²−0.5` fires iff `A≠B`. Two strawmen — "
        "zero the attention, or use a linear readout — collapse to the floor."
    )

    with gr.Tabs():
        with gr.Tab("Demo"):
            gr.Markdown("### Full XOR truth table vs. ablations")
            table = gr.Markdown(truth_table())
            gr.Markdown("### Latest run ablation accuracies (from main.py)")
            ablate_md = gr.Markdown(latest_ablation_md())
            refresh = gr.Button("Refresh")
            refresh.click(
                fn=lambda: (truth_table(), latest_ablation_md()),
                inputs=None,
                outputs=[table, ablate_md],
            )
            demo.load(
                fn=lambda: (truth_table(), latest_ablation_md()),
                inputs=None,
                outputs=[table, ablate_md],
            )

        with gr.Tab("Benchmark"):
            benchmark_panel(GOAL_DIR)

if __name__ == "__main__":
    demo.launch()
