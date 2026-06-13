"""Demo + Benchmark app for the wildcard n-gram attention head (pass_4).

Demo tab tells the whole story from a single run's `comparison.json`:
  * a sharpness/anchor-mass sweep over wildcard span comparing the hand-built
    circuit against a positional strawman, an ablated control, and the uniform
    baseline — showing the circuit alone stays high & flat as the gap widens;
  * a per-position attention bar (the target's mean attention row) so you can
    literally see the mass land on the anchor and skip the wildcards.
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = Path(__file__).resolve().parent
GOAL_DIR = ATTEMPT_DIR.parent
RESULTS = ATTEMPT_DIR / "results"

VARIANT_COLORS = {"circuit": "#1b7837", "prev_token": "#c2521a", "ablated": "#7a7a7a"}
VARIANT_LABELS = {
    "circuit": "circuit (hand-built matcher)",
    "prev_token": "prev-token strawman",
    "ablated": "ablated (match weight zeroed)",
}


def list_runs():
    if not RESULTS.exists():
        return []
    runs = [p.name for p in RESULTS.iterdir() if (p / "comparison.json").exists()]
    return sorted(runs, reverse=True)


def load_comparison(run_id):
    if not run_id:
        return None
    f = RESULTS / run_id / "comparison.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def sweep_figure(run_id):
    comp = load_comparison(run_id)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.0))
    if comp is None:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "no run data", ha="center", va="center")
            ax.set_axis_off()
        return fig
    spans = comp["spans"]

    # Top: anchor mass (the directly interpretable "does B attend to A?").
    for name, recs in comp["variants"].items():
        ax1.plot(spans, [r["mean_anchor"] for r in recs], marker="o",
                 color=VARIANT_COLORS[name], label=VARIANT_LABELS[name])
    ax1.plot(spans, comp["uniform_baseline_anchor_mass"], linestyle="--",
             color="black", label="uniform baseline")
    ax1.set_ylim(-0.05, 1.08)
    ax1.set_ylabel("mean target→anchor attention")
    ax1.set_xlabel("wildcard span  k   (A " + "* " * 1 + "… B)")
    ax1.set_xticks(spans)
    ax1.set_title("Does the target attend to the anchor as the gap widens?")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="center right", fontsize=8)

    # Bottom: sharpness (the scored metric), log scale.
    for name, recs in comp["variants"].items():
        ys = [max(r["sharpness"], 1e-3) for r in recs]
        ax2.plot(spans, ys, marker="s", color=VARIANT_COLORS[name],
                 label=VARIANT_LABELS[name])
    ax2.plot(spans, [max(x, 1e-3) for x in comp["uniform_baseline_sharpness"]],
             linestyle="--", color="black", label="uniform baseline")
    ax2.set_yscale("log")
    ax2.set_ylabel("sharpness  (scored metric, log)")
    ax2.set_xlabel("wildcard span  k")
    ax2.set_xticks(spans)
    ax2.set_title("Sharpness = anchor / (wildcard + other).  Circuit stays flat ≈ clean skip.")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="center right", fontsize=8)

    fig.tight_layout()
    return fig


def row_figure(run_id, variant, span):
    comp = load_comparison(run_id)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    if comp is None or variant not in comp["variants"]:
        ax.text(0.5, 0.5, "no run data", ha="center", va="center")
        ax.set_axis_off()
        return fig
    span = int(span)
    recs = comp["variants"][variant]
    rec = next((r for r in recs if r["span"] == span), None)
    if rec is None:
        ax.text(0.5, 0.5, "span not in run", ha="center", va="center")
        ax.set_axis_off()
        return fig

    row = rec["mean_row"]
    L = len(row)
    anchor_pos = comp.get("anchor_pos", 0)
    target_pos = rec["target_pos"]
    wild = set(range(1, 1 + span))
    colors = []
    for j in range(L):
        if j == anchor_pos:
            colors.append("#1b7837")        # anchor (green)
        elif j in wild:
            colors.append("#c2521a")        # wildcard (orange)
        elif j == target_pos:
            colors.append("#444444")        # target/self (dark)
        else:
            colors.append("#c9c9c9")        # filler (light grey)
    ax.bar(range(L), row, color=colors)
    ax.set_xticks(range(L))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("key position  (green=anchor, orange=wildcard, dark=target/self, grey=filler)")
    ax.set_ylabel("mean attention")
    ax.set_title(f"{VARIANT_LABELS[variant]} — target row, span k={span} "
                 f"(target at pos {target_pos})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


with gr.Blocks() as demo:
    gr.Markdown(
        "# Wildcard n-gram attention — `pass_4` (hand-built head)\n"
        "A single attention head with a **hand-set Q/K circuit**: the *target* "
        "token's query and the *anchor* token's key share one feature channel, "
        "so the target attends back to the anchor by **token identity** and "
        "ignores whatever wildcard tokens sit between them.\n\n"
        "The claim is testable against three controls: a **prev-token** "
        "positional strawman (bigram head — can't skip), an **ablated** circuit "
        "(matching weight zeroed → uniform), and the **uniform baseline**."
    )

    with gr.Tab("Demo"):
        with gr.Row():
            run_dd = gr.Dropdown(choices=list_runs(), value=(list_runs()[0] if list_runs() else None),
                                 label="Run", scale=3)
            refresh = gr.Button("↻ Refresh runs", scale=1)

        sweep_plot = gr.Plot(label="Skip robustness across wildcard span")

        gr.Markdown("### Look inside one condition")
        with gr.Row():
            variant_dd = gr.Dropdown(
                choices=list(VARIANT_LABELS.keys()), value="circuit",
                label="Variant", scale=2,
            )
            span_sl = gr.Slider(0, 4, step=1, value=1, label="Wildcard span k", scale=2)
        row_plot = gr.Plot(label="Target's mean attention over key positions")

        def _refresh():
            runs = list_runs()
            sel = runs[0] if runs else None
            return gr.update(choices=runs, value=sel)

        def _update_sweep(run_id):
            return sweep_figure(run_id)

        def _update_row(run_id, variant, span):
            return row_figure(run_id, variant, span)

        refresh.click(_refresh, outputs=run_dd)
        run_dd.change(_update_sweep, inputs=run_dd, outputs=sweep_plot)
        run_dd.change(_update_row, inputs=[run_dd, variant_dd, span_sl], outputs=row_plot)
        variant_dd.change(_update_row, inputs=[run_dd, variant_dd, span_sl], outputs=row_plot)
        span_sl.change(_update_row, inputs=[run_dd, variant_dd, span_sl], outputs=row_plot)

        demo.load(_update_sweep, inputs=run_dd, outputs=sweep_plot)
        demo.load(_update_row, inputs=[run_dd, variant_dd, span_sl], outputs=row_plot)

    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
