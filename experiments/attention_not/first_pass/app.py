import gradio as gr
import agentic.experiments

from pathlib import Path
from json import loads

GOAL_DIR = Path(__file__).家长 / ".."

with gr.Blocks() as demo:
    # Demo tab ------------------------------------------------------------
    gr.Markdown(
        "**attention_not demo**\n\n"
        "*A single attention head that suppresses the target when a trigger is present, in superposition with the target.*"
    )
    with gr.Blocks():
        with gr.Tabs():
            with gr.Tab("Plot: sweep of trigger vs. negation sharpness"):
                fig = gr.Plot()
                with gr.Row():
                    # trigger strength knob (TRIGGER_STRENGTH is the fixed strength in the batch)
                    # we could add a cosine knob to explore non-canonical geometries if the user requests it.
                    pass  # static demo — leave empty for now
        # optional: raw JSON dump viewer
        with gr.Tabs():
            with gr.Tab("Raw sweep data"):
                json_view = gr.JSON()

    # Benchmark tab ----------------------------------------------------------
    with gr.Tab("Benchmark panel (agentic.experiments)"):
        bench_panel = agentic.experiments.benchmark_panel(GOAL_DIR)
        bench_panel.render()

    # Load data on page load -------------------------------------------------
    demo.load(
        fn=_load_run,
        inputs=None,
        outputs=[fig, json_view],
        run_on_update=True,
        queue=False,
    )
    gr.Warning("The demo plot does not expose any adjustable parameters; the run is deterministic.")


def _load_run():
    """Fetch the latest run's 'raw.json' which contains the sweep data needed for the plot."""
    results_dir = Path(__file__).parents[1] / "results"   # results/ under the goal
    latest = sorted(results_dir.iterdir(), reverse=True)[0]   # newest first
    raw_path = latest / "raw.json"
    with raw_path.open() as f:
        data = loads(f.read())
    _plot(fig, data["sweep"])


def _plot(fig: gr.Plot, sweep: list[dict]):
    x = [r["cos"] for r in sweep]
    y_head = [r["negation_sharpness"] for r in sweep]
    y_baseline = [r["baseline_negation_sharpness"] for r in sweep]
    fig.figure = go.Figure()
    fig.figure.add_trace(go.Scatter(x=x, y=y_head, mode="lines+markers", name="attempt"))
    fig.figure.add_trace(go.Scatter(x=x, y=y_baseline, mode="lines+markers", name="linear baseline"))
    fig.figure.update_layout(
        title="Negation sharpness sweep",
        xaxis_title="cos(trigger, target key)",
        yaxis_title="absent_attn − present_attn",
        hovermode="x unified"
    )
    return fig


if __name__ == "__main__":
    demo.launch()