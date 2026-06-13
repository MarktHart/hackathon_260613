import gradio as gr
import numpy as np
import json
from pathlib import Path
from agentic.experiments import benchmark_panel
from agentic.experiments.load_task import loadTask  # For introspecting task.py

# ============ Load task ============
task = loadTask(__file__, "task.py")
# Get static metadata
SEQ_LEN = task._task.seq_len   # type: ignore[attr-defined]
MODES = task._task.modes     # type: ignore[attr-defined]

# ============ Helper: load payload from latest run ============
def _latest_run() -> tuple[Path, dict]:
    goal_dir = Path(__file__).parent.parent
    run_dir = next(goal_dir.glob("results/*"), None)
    if not run_dir:
        raise FileNotFoundError(f"No run found under {goal_dir}/results/")
    payload_path = run_dir / "benchmark.json"
    if not payload_path.is_file():
        raise FileNotFoundError(f"No benchmark.json at {payload_path}")
    with payload_path.open() as f:
        payload = json.load(f)
    return run_dir, payload

run_dir, payload = _latest_run()

# ============ Demo tab ============
with gr.Blocks() as demo:
    gr.Markdown(f"# Heuristic attention-mode classifier (first_pass)")

    alphaSlider = gr.Slider(
        value=10.0,
        minimum=0.1,
        maximum=100.0,
        step=0.1,
        label="Noise concentration (α)",
        interactive=False,   # will be updated via sweep selection
    )
    patternImg = gr.Image(height=256, width=256, show_label=False)
    predLabel = gr.Label(num_top_classes=1, label="Predicted mode")
    gTruthLabel = gr.Label(num_top_classes=1, label="Ground-truth mode")

    # Dropdown of α values from the sweep
    sweepSelect = gr.Dropdown(
        choices=[f"α={_fmt_v(a)}" for a in payload["sweep"]],
        label="Select sweep point",
        interactive=True,
        value="α=10",   # canonical
    )

    # Choose a random pattern from the selected sweep batch
    def _select_pattern(sweep_idx: gr.Dropdown):
        batch = payload["sweep"][sweep_idx]
        alpha = batch["alpha"]
        # Pick random index in batch.labels
        i = np.random.randint(0, len(batch["logits"]))
        A = batch["patterns"][i]
        label = batch["labels"][i]
        # Heuristic: we don't store raw patterns, but we can reconstruct a synthetic view
        # for demonstration — just show the actual `A` as heatmap.
        return {
            "alphaSlider": float(alpha),
            "patternImg": (None, None, np.array(A), "HEATMAP"),   # gr.Image expects tuple
            "predLabel": [(j, float(batch["logits"][i][j])) for j in range(len(MODES))],
            "gTruthLabel": [(label, MODES[label])],
        }

    sweepSelect.change(
        fn=_select_pattern,
        inputs=[sweepSelect],
        outputs=[
            alphaSlider,
            patternImg,
            predLabel,
            gTruthLabel,
        ],
    )

    # Initial load
    demo.load(
        fn=_select_pattern,
        inputs=[gr.Number(value=3)],   # pick one of the 7 batches ( canonical α=10 is index 2)
        outputs=[alphaSlider, patternImg, predLabel, gTruthLabel],
    )


# ============ Benchmark tab ============
with gr.Blocks() as benchmark:
    gr.Markdown("## Benchmark across all attempts")
    benchmark_panel("attention_mode")   # scans experiments/attention_mode/


# ============ Serve ============
# The pipeline checks the module-level demo attribute
demo = demo   # keep for import check (redundant but safe)
if __name__ == "__main__":
    demo.launch()


# ============ Utility ============
def _fmt_v(v):
    if float(v).is_integer():
        return str(int(v))
    return f"{v:g}".replace(".", "p")