import gradio as gr

from agentic.experiments import load_task, benchmark_panel

from pathlib import Path


def _demo_table(sweep):
    """Render a readable table of the sweep (levels 0..4)."""
    rows = [["Level", "Block size", "Uniform baseline", "Best head mass", "Best head purity"]]
    for rec in sweep:
        h = rec["level"]
        u = rec["uniform_mass"]
        mass = rec["best_head_mass"]
        purity = (mass - u) / (1 - u) if 1 - u > 0 else 0
        rows.append([h, rec["block_size"], f"{u:.2f}", f"{mass:.2f}", f"{purity:.2f}"])
    return rows


def _demo_run():
    import json
    from collections import OrderedDict

    payload = json.loads(_demo_payload)
    return _demo_table(payload["sweep"])


try:
    _demo_payload = Path("results/").rglob("benchmark.json").sorted()[-1].read_text()
except FileNotFoundError:
    raise RuntimeError("No benchmark.json found in results/ — did you run main.py first?")


with gr.Blocks() as demo:
    gr.Markdown("# attention_hierarchical_pool – first_pass")
    gr.Markdown(
        "**What's being shown**: a hand-built single-layer transformer of 8 heads."
        "\nEach head `h` (0..4) is a pure within-block pooler at level `h` of a binary hierarchy."
        "\nHeads 5..7 are dummy uniform heads."
    )
    output = gr.Table(
        label="Sweep over hierarchy levels (0..4)",
        columns=["Level", "Block size", "Uniform baseline (mass)",
                 "Best head mass", "Best head purity"],
    )
    gr.Markdown("## Run the demo")
    with gr.Row(equal_size="sm"):
        with gr.Column(scale=3):
            gr.Markdown(
                "The demo simply loads the most recent `benchmark.json` from `results/` "
                "and prints the sweep. Refresh the page after running `main.py`. "
                "This shows that the hand-built heads *do* pool at their assigned levels, "
                "and that the purity rises with larger block sizes."
            )
        with gr.Column(scale=1):
            btn = gr.Button("Refresh", variant="primary")
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("## Benchmark (all attempts at this goal)")
            bench = benchmark_panel("../..")
    with gr.Column(scale=1):
        # dummy button to keep the demo from being empty
        pass

    btn.click(_demo_run, [], output)

if __name__ == "__main__":
    demo.launch()