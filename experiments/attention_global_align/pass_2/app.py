"""Gradio app for attention_global_align / pass_2.

Demo tab: the alignment-vs-interference curve for the tempered retrieval head,
overlaid with three strawmen (raw K@q, temperature=0 ablation, random logits)
and the uniform baseline. A second panel shows, at a chosen interference
slice, how the mass splits between target and distractor.

Benchmark tab: the shared cross-attempt leaderboard / history panel.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = Path(__file__).resolve().parent
GOAL_DIR = ATTEMPT_DIR.parent
RESULTS_DIR = ATTEMPT_DIR / "results"


def _run_ids() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    runs = [p.name for p in RESULTS_DIR.iterdir()
            if p.is_dir() and (p / "comparison.json").exists()]
    return sorted(runs, reverse=True)


def _load(run_id: str) -> dict | None:
    if not run_id:
        return None
    fp = RESULTS_DIR / run_id / "comparison.json"
    if not fp.exists():
        return None
    with open(fp) as fh:
        return json.load(fh)


def _curve_fig(run_id: str):
    data = _load(run_id)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if data is None:
        ax.text(0.5, 0.5, "no run selected", ha="center", va="center")
        return fig

    cos = data["distractor_cos_sweep"]
    styles = {
        "tempered head": dict(color="#1b7837", lw=3, marker="o", zorder=5),
        "raw K@q": dict(color="#b35806", lw=2, marker="s", ls="--"),
        "temperature=0": dict(color="#7b3294", lw=2, marker="^", ls=":"),
        "random logits": dict(color="#999999", lw=1.6, marker="x", ls=":"),
    }
    for name, curve in data["variants"].items():
        key = next((k for k in styles if name.startswith(k)), None)
        st = styles.get(key, {})
        ax.plot(cos, curve, label=name, **st)

    ax.plot(cos, data["uniform_baseline"], color="black", lw=1.4, ls="-.",
            label="uniform baseline (1/L)")
    ax.axhline(data["robustness_ceiling"], color="#1b7837", lw=1, ls="--",
               alpha=0.5)
    ax.text(0.02, data["robustness_ceiling"] + 0.01,
            "0.5 = ceiling at cos=1 (distractor == target)",
            color="#1b7837", fontsize=8)

    ax.axvline(data["canonical_cos"], color="red", lw=1, alpha=0.4)
    ax.set_xlabel("distractor cosine to target  (interference →)")
    ax.set_ylabel("global alignment = attention mass on target")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Global alignment vs interference (β=%g)" % data["beta"])
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _split_fig(run_id: str, slice_idx: int):
    data = _load(run_id)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if data is None:
        ax.text(0.5, 0.5, "no run selected", ha="center", va="center")
        return fig

    cos = data["distractor_cos_sweep"]
    i = max(0, min(int(slice_idx), len(cos) - 1))
    ours_name = next(k for k in data["variants"] if k.startswith("tempered"))
    target = data["variants"][ours_name][i]
    distr = data["ours_distractor_mass"][i]
    other = max(0.0, 1.0 - target - distr)

    bars = ax.bar(["target", "distractor", "other keys"],
                  [target, distr, other],
                  color=["#1b7837", "#b35806", "#cccccc"])
    for b, v in zip(bars, [target, distr, other]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("attention mass")
    ax.set_title("Tempered head: mass split at distractor cos = %g" % cos[i])
    fig.tight_layout()
    return fig


def _summary(run_id: str) -> str:
    data = _load(run_id)
    if data is None:
        return "No run found. Run `main.py` first."
    cos = data["distractor_cos_sweep"]
    ci = cos.index(data["canonical_cos"])
    ours_name = next(k for k in data["variants"] if k.startswith("tempered"))
    ours = data["variants"][ours_name]
    raw = data["variants"]["raw K@q (beta=1)"]
    uni = data["uniform_baseline"]
    rob = ours[-1] / max(ours[0], 1e-9)
    return (
        f"**β = {data['beta']:g}**  |  "
        f"canonical (cos=0.5) alignment **{ours[ci]:.3f}** "
        f"vs raw K@q {raw[ci]:.3f} vs uniform {uni[ci]:.3f}  \n"
        f"robustness = align(cos=1)/align(cos=0) = "
        f"{ours[-1]:.3f}/{ours[0]:.3f} = **{rob:.3f}**  "
        f"(0.5 is the physical ceiling — at cos=1 the distractor key is "
        f"identical to the target, so mass can at best split 50/50)."
    )


with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_global_align — tempered retrieval head\n"
        "`base_model.py` attention + one delta: a temperature β on `K@q`. "
        "Sharpening turns weak, smeared alignment into near-perfect retrieval "
        "until the distractor becomes mathematically identical to the target."
    )
    with gr.Tabs():
        with gr.TabItem("Demo"):
            with gr.Row():
                run_dd = gr.Dropdown(
                    choices=_run_ids(),
                    value=(_run_ids()[0] if _run_ids() else None),
                    label="run",
                )
            summary_md = gr.Markdown()
            with gr.Row():
                curve_plot = gr.Plot(label="alignment vs interference")
            slice_sl = gr.Slider(0, 4, step=1, value=2,
                                 label="interference slice index (0=cos0 … 4=cos1)")
            split_plot = gr.Plot(label="mass split at slice")

            def _refresh(run_id, slc):
                return (_curve_fig(run_id), _split_fig(run_id, slc),
                        _summary(run_id))

            run_dd.change(_refresh, [run_dd, slice_sl],
                          [curve_plot, split_plot, summary_md])
            slice_sl.change(_refresh, [run_dd, slice_sl],
                            [curve_plot, split_plot, summary_md])
            demo.load(_refresh, [run_dd, slice_sl],
                      [curve_plot, split_plot, summary_md])

        with gr.TabItem("Benchmark"):
            benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
