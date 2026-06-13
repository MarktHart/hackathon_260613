import gradio as gr
from pathlib import Path
import json
import numpy as np
import torch
from agentic.experiments import benchmark_panel
from agentic.experiments.load_task import loadTask


# ============ Load task ============
task = loadTask(__file__, "task.py")
L = getattr(task, "L", 16)                # L = 16
MODES = getattr(task, "modes", [])
assert isinstance(MODES, (list, tuple)), "modes must be list/tuple"
N_PER_MODE = getattr(task, "N_PER_MODE", 10)
NOISE_LEVELS = getattr(task, "NOISE_LEVELS", [0.0, 0.1, 0.2, 0.3, 0.5])


# ============ Helper: load latest payload ============
def _latest_run() -> tuple[Path, dict]:
    goal_dir = Path(__file__).parent.parent
    results_path = goal_dir / "results"
    if not results_path.is_dir():
        raise FileNotFoundError(f"No results/ directory under {goal_dir}")
    latest_run = sorted(results_path.iterdir())[-1]
    payload_path = latest_run / "benchmark.json"
    if not payload_path.is_file():
        raise FileNotFoundError(f"No benchmark.json at {payload_path}")
    with payload_path.open() as f:
        payload = json.load(f)
    return latest_run, payload


# ============ Demo tab ============
with gr.Blocks() as demo:
    gr.Markdown("# Attention-Mode Classifier (pass_3)\n"
                  "Classifies a 1D attention pattern using KL-min divergence to precomputed clean matrices."
                  )

    # Head index selection
    head_select = gr.Dropdown(
        choices=[f"head {i}" for i in range(10 * 5)],   # total N_PER_MODE * N_MODES = 50
        label="Select head",
        value="head 0",
        interactive=True,
    )

    # Ground Truth Label
    gtruth = gr.Label(num_top_classes=1, label="Ground truth mode")
    # Model Prediction Bar
    pred = gr.Label(num_top_classes=5, label="Predicted probabilities")

    # Visual: lower-triangular pattern heatmap
    pattern_img = gr.Image(height=L+20, width=L+20, show_label=False)
    # KL values summary (lower is better)
    kl_detail = gr.Textbox(label="KL scores (per-mode, lower better)", interactive=False)

    # Load data once
    run_dir, payload = _latest_run()
    sweep = payload["sweep"]   # list of dicts


    def _build_pattern_heatmap(data: dict) -> np.ndarray:
        # For visualisation, we render the head's actual pattern as a
        # lower-triangular heatmap (L x L). We use the mode name to colour
        # a reference clean matrix (just for UI; not used in scoring).
        noise = data["noise"]
        true_mode = data["true_mode"]
        # Build a colour matrix: rows are attention distributions.
        # Use a single colour channel for gr.Image heatmap support.
        mat = data["attention_matrices"]   # (L, L) float32
        A = mat.copy().astype(np.float32)
        # gr.Image expects (H, W, C=1) for single-channel heatmaps.
        return A


    def _make_pattern_image(data: dict) -> dict:
        A = _build_pattern_heatmap(data)
        # Return (H, W, 1) for Gradio heatmap.
        return (None, None, np.dstack([A]), "HEATMAP")


    def _on_head_select(sweep_idx):
        idx = int(sweep_idx.split(" ")[1])
        data = sweep[idx]
        true_mode = data["true_mode"]
        pred_probs = data["pred_probs"]
        kl_msg = ""
        # Simulate the classifier by hand (KL per mode, then softmax).
        # Not stored in payload; compute quickly for UI only.
        L_local = task.L
        # Reference clean matrices (simple)
        clean = {
            "positional": np.eye(L_local, dtype=np.float32)[:, 0],
            "uniform": np.full(L_local, 1.0 / L_local, dtype=np.float32),
            "diagonal": np.eye(L_local).diagonal(),
            "induction": np.eye(L_local, dtype=np.float32).diagflat(offset=1).sum(axis=1),
            "previous_token": np.eye(L_local, dtype=np.float32).diagflat(offset=-1).sum(axis=1),
        }
        kls = []
        for m in MODES:
            # Approx: sum of KL from pattern to clean matrix (per query).
            row_kls = []
            for q in range(L_local):
                row_pattern = data["attention_matrices"][q]
                row_clean = (clean[m] / clean[m].sum())
                row_kls.append(kl_pq(row_pattern, row_clean))
            kls.append(np.mean(row_kls))
        kl_msg = f"KL scores: {dict(zip(MODES, kls))}"
        # Return UI values.
        return {
            "pattern_img": _make_pattern_image(data),
            "gtruth": [(true_mode, 1.0)],
            "pred": [(m, pred_probs[m]) for m in MODES],
            "kl_detail": kl_msg,
        }


    head_select.change(
        fn=_on_head_select,
        inputs=[head_select],
        outputs=[pattern_img, gtruth, pred, kl_detail],
    )

    demo.load(
        fn=lambda: _on_head_select("head 0"),   # default first head on load
        inputs=[],
        outputs=[pattern_img, gtruth, pred, kl_detail],
    )


# ============ Benchmark leaderboard ============
with gr.Blocks() as benchmark:
    gr.Markdown("## Benchmark across all attempts")
    benchmark_panel("attention_mode")   # scans experiments/attention_mode/


# ============ Export for boot-check ============
demo = demo   # expose module-level demo: gr.Blocks

if __name__ == "__main__":
    demo.launch()