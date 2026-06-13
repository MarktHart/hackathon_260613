"""Gradio app: per-sequence content->position pointer for k-th selection.

Demo tab views (radio):
  1. Sweep accuracy   -- attn_at_k vs k for every method, with the analytic
     Bayes ceiling and the uniform baseline. The faithful pointer tracks k and
     sits on the ceiling; the fixed-position strawman only nails k=8; the
     cross-sequence oracle (unfaithful) reaches 1.0 everywhere.
  2. Attention by k   -- mean attention over positions for a chosen k.
  3. Operating range  -- pointer accuracy vs spurious-marker rate r, empirical
     vs analytic, showing exactly where the mechanism degrades.
  4. Ablation         -- knocking out the query / marker channel collapses the
     head to the uniform baseline (causal evidence).

Benchmark tab: cross-attempt leaderboard via benchmark_panel.
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"

METHOD_STYLE = {
    "content_pointer": ("#1f77b4", "content pointer (faithful, per-seq)"),
    "fixed_position@8": ("#d62728", "fixed-position@8 (strawman)"),
    "oracle_batch": ("#2ca02c", "cross-seq oracle (UNFAITHFUL)"),
    "uniform": ("#7f7f7f", "uniform baseline"),
}


def _list_runs() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    runs = [p.name for p in RESULTS_DIR.iterdir()
            if p.is_dir() and (p / "comparison.json").exists()]
    return sorted(runs, reverse=True)


def _load(run: str):
    if not run:
        return None
    path = RESULTS_DIR / run / "comparison.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _ks_for_run(run: str) -> list[int]:
    comp = _load(run)
    return comp.get("sweep_k", [8]) if comp else [8]


def _blank(msg: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center")
    ax.axis("off")
    return fig


def _rec_at_k(recs, k):
    return next((r for r in recs if r["k"] == k), None)


def render(run: str, view: str, k: int):
    comp = _load(run)
    if comp is None:
        return _blank("no run found - execute main.py first"), "No data."

    L = comp["L"]
    ceiling = comp["analytic_ceiling"]
    base = comp["uniform_baseline"]

    if view == "Sweep accuracy":
        fig, ax = plt.subplots(figsize=(8, 4.4))
        ks = comp["sweep_k"]
        for name, (color, label) in METHOD_STYLE.items():
            recs = comp["methods"].get(name, [])
            ys = [(_rec_at_k(recs, kk) or {}).get("attn_at_k", 0.0) for kk in ks]
            ax.plot(ks, ys, color=color, marker="o", label=label)
        ax.axhline(ceiling, color="black", ls="--", lw=1,
                   label=f"Bayes ceiling = {ceiling:.3f}")
        ax.axhline(base, color="grey", ls=":", lw=1,
                   label=f"uniform 1/L = {base:.3f}")
        ax.set_xlabel("target position k")
        ax.set_ylabel("attn_at_k (mass on correct position)")
        ax.set_title("k-th selection accuracy across the sweep")
        ax.set_ylim(-0.03, 1.05)
        ax.legend(fontsize=8, loc="center right")
        fig.tight_layout()
        cp = comp["methods"]["content_pointer"]
        mean_cp = sum(r["attn_at_k"] for r in cp) / len(cp)
        txt = (f"Faithful pointer mean attn_at_k = {mean_cp:.3f}  "
               f"(analytic ceiling {ceiling:.3f}).\n"
               "It tracks every k; the fixed-position strawman only spikes at "
               "k=8; the oracle hits 1.0 by aggregating across sequences -- "
               "something a real head cannot do.")
        return fig, txt

    if view == "Attention by k":
        fig, ax = plt.subplots(figsize=(8, 4.4))
        lines = [f"Target position k = {k}", ""]
        for name in ("content_pointer", "oracle_batch", "uniform"):
            color, label = METHOD_STYLE[name]
            rec = _rec_at_k(comp["methods"].get(name, []), k)
            if rec is None:
                continue
            ax.plot(range(L), rec["mean_attn"], color=color, marker=".", label=label)
            lines.append(f"{label}: attn@k={rec['attn_at_k']:.3f} "
                         f"sharpness={rec['sharpness']:.3f}")
        ax.axvline(k, color="black", ls="--", lw=1, label=f"true k={k}")
        ax.set_xlabel("position in sequence")
        ax.set_ylabel("mean attention weight")
        ax.set_title("Mean attention over positions")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        return fig, "\n".join(lines)

    if view == "Operating range":
        fig, ax = plt.subplots(figsize=(8, 4.4))
        rows = comp["operating_range"]
        rs = [d["r"] for d in rows]
        ax.plot(rs, [d["empirical_acc"] for d in rows], color="#1f77b4",
                marker="o", label="content pointer (empirical)")
        ax.plot(rs, [d["analytic_acc"] for d in rows], color="black", ls="--",
                marker="x", label="Bayes-optimal E[1/(1+S)]")
        ax.axvline(1.0 / comp["V"], color="grey", ls=":",
                   label=f"task rate r=1/V={1.0/comp['V']:.3f}")
        ax.set_xlabel("spurious-marker rate r  (P other position carries M)")
        ax.set_ylabel("attn_at_k at canonical k=8")
        ax.set_title("Operating range: where the mechanism degrades")
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=8)
        fig.tight_layout()
        txt = ("Empirical accuracy sits exactly on the analytic ceiling across "
               "two orders of magnitude of r. Accuracy falls as collisions rise "
               "-- the irreducible per-sequence ambiguity, predicted in closed "
               "form, not a bug.")
        return fig, txt

    # Ablation
    fig, ax = plt.subplots(figsize=(8, 4.4))
    abl = comp["ablation"]
    order = ["content_pointer", "query_ablated", "marker_channel_ablated", "uniform_baseline"]
    labels = ["pointer\n(intact)", "query\nablated", "marker chan\nablated", "uniform\nbaseline"]
    vals = [abl[k] for k in order]
    colors = ["#1f77b4", "#d62728", "#d62728", "#7f7f7f"]
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("attn_at_k at canonical k=8")
    ax.set_title("Causal ablation of the hand-built circuit")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    txt = ("Zeroing the marker query, or zeroing the marker channel of the keys, "
           "collapses the head to the uniform baseline (1/L). The selection "
           "behaviour is caused specifically by the marker-keyed query.")
    return fig, txt


with gr.Blocks(title="attention_kth_select / pass_2") as demo:
    gr.Markdown(
        "# k-th position selection - faithful per-sequence content pointer\n"
        "A single hand-set attention head keys on the marker **within each "
        "sequence** (no cross-sequence aggregation). Its accuracy hits the "
        "**Bayes-optimal ceiling** set by spurious-marker collisions (~0.86), "
        "not a fake 1.0. The unfaithful cross-sequence oracle is shown only for "
        "contrast."
    )

    runs = _list_runs()
    default_run = runs[0] if runs else ""
    default_ks = _ks_for_run(default_run)

    with gr.Tab("Demo"):
        with gr.Row():
            run_dd = gr.Dropdown(choices=runs, value=default_run, label="run", scale=2)
            view_dd = gr.Radio(
                choices=["Sweep accuracy", "Attention by k", "Operating range", "Ablation"],
                value="Sweep accuracy", label="view", scale=3,
            )
            k_dd = gr.Dropdown(
                choices=default_ks,
                value=8 if 8 in default_ks else (default_ks[0] if default_ks else 8),
                label="k (for 'Attention by k')", scale=1,
            )
        plot = gr.Plot(label="result")
        summary = gr.Textbox(label="summary", lines=4)

        def _refresh(run, view, k):
            return render(run, view, k)

        run_dd.change(
            lambda r: gr.update(choices=_ks_for_run(r),
                                value=8 if 8 in _ks_for_run(r) else _ks_for_run(r)[0]),
            inputs=run_dd, outputs=k_dd,
        )
        run_dd.change(_refresh, inputs=[run_dd, view_dd, k_dd], outputs=[plot, summary])
        view_dd.change(_refresh, inputs=[run_dd, view_dd, k_dd], outputs=[plot, summary])
        k_dd.change(_refresh, inputs=[run_dd, view_dd, k_dd], outputs=[plot, summary])
        demo.load(_refresh, inputs=[run_dd, view_dd, k_dd], outputs=[plot, summary])

    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
