"""Gradio app for pass_2.

Demo tab tells the whole story without the README:

  (A) Faithfulness curve — pairwise F1 vs ATTENTION DEPTH, one line per
      component diameter, with the adjacency (1-hop) baseline overlaid. The
      model curve sits on the baseline at depth 1 and jumps to F1=1.0 exactly
      at depth == diameter: removing attention hops causally collapses the
      closure back to the strawman.

  (B) Per-layer reachability heatmaps — pick a diameter and slide the attention
      depth; watch the predicted same-component map fill in hop-by-hop until it
      matches ground truth. Errors are colour-coded against the truth.

Benchmark tab drops in the shared cross-attempt panel.

Pure numpy/pandas/gradio — reads artefacts written by main.py, no torch/GPU at
import or interaction time, so the boot-check is safe.
"""
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"


# --------------------------------------------------------------------------- #
# Artefact loading
# --------------------------------------------------------------------------- #
def _run_dirs():
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        [p for p in RESULTS_DIR.glob("*") if (p / "ablation.json").exists()]
    )


def _run_choices():
    dirs = _run_dirs()
    return [p.name for p in dirs]


def _latest_run_name():
    choices = _run_choices()
    return choices[-1] if choices else None


def _load_ablation(run_name):
    if run_name is None:
        return None
    path = RESULTS_DIR / run_name / "ablation.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_reach(run_name, diameter):
    path = RESULTS_DIR / run_name / f"reach_diam_{int(diameter)}.npz"
    if not path.exists():
        return None
    return np.load(path)


# --------------------------------------------------------------------------- #
# (A) Faithfulness curve
# --------------------------------------------------------------------------- #
def _curve_df(ablation):
    if ablation is None:
        return pd.DataFrame({"depth": [], "f1": [], "series": []})
    depths = ablation["depths"]
    rows = []
    for D in ablation["diameters"]:
        ys = ablation["model_f1"][str(D)]
        for depth, y in zip(depths, ys):
            rows.append({"depth": int(depth), "f1": float(y),
                         "series": f"attention·diam{D}"})
        b = float(ablation["baseline_f1"][str(D)])
        for depth in depths:
            rows.append({"depth": int(depth), "f1": b,
                         "series": f"adj-baseline·diam{D}"})
    return pd.DataFrame(rows)


def _headline_md(ablation):
    if ablation is None:
        return "_No run found. Execute `main.py` first._"
    canon = ablation["canonical_diameter"]
    head = ablation["transitive_closure_robustness"]
    lines = [
        f"### Attention recovers the connected components",
        f"**transitive_closure_robustness (headline)** = "
        f"`{head:.3f}`  ·  full-depth attention solves every diameter.",
        "",
        "| diameter | attention F1 (full depth) | adjacency baseline F1 | lift |",
        "|---|---|---|---|",
    ]
    for D in ablation["diameters"]:
        m = float(ablation["full_depth_f1"][str(D)])
        b = float(ablation["baseline_f1"][str(D)])
        star = " ⟵ canonical" if int(D) == int(canon) else ""
        lines.append(f"| {D} | {m:.3f} | {b:.3f} | {m - b:+.3f}{star} |")
    lines += [
        "",
        "At **depth 1** the attention read-out equals the 1-hop adjacency "
        "baseline; each extra attention hop adds one hop of reach. The curve "
        "jumps to F1=1.0 exactly at **depth == diameter** — knock hops out and "
        "the closure collapses back to adjacency (the built-in ablation).",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# (B) Reachability heatmaps (numpy -> RGB, no matplotlib)
# --------------------------------------------------------------------------- #
def _order(labels):
    return np.argsort(labels, kind="stable")


def _upscale(rgb, scale=18):
    return np.kron(rgb, np.ones((scale, scale, 1), dtype=np.uint8)).astype(np.uint8)


def _truth_image(truth, labels):
    order = _order(labels)
    t = truth[np.ix_(order, order)].astype(bool)
    n = t.shape[0]
    img = np.full((n, n, 3), 255, dtype=np.uint8)
    img[t] = (40, 80, 170)          # same component -> navy
    di = np.arange(n)
    img[di, di] = (60, 60, 60)      # diagonal -> grey
    return _upscale(img)


def _pred_image(pred, truth, labels):
    order = _order(labels)
    p = pred[np.ix_(order, order)].astype(bool)
    t = truth[np.ix_(order, order)].astype(bool)
    n = p.shape[0]
    img = np.full((n, n, 3), 255, dtype=np.uint8)
    img[p & t] = (40, 160, 70)      # TP  -> green
    img[~p & t] = (240, 150, 30)    # FN  -> orange (missed same-component pair)
    img[p & ~t] = (210, 50, 50)     # FP  -> red (false merge)
    di = np.arange(n)
    img[di, di] = (60, 60, 60)
    return _upscale(img)


def _f1_offdiag(pred, truth):
    n = pred.shape[0]
    iu = np.triu_indices(n, k=1)
    p = pred[iu].astype(bool)
    t = truth[iu].astype(bool)
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom else 0.0


def _render_heatmaps(run_name, diameter, depth):
    npz = _load_reach(run_name, diameter)
    if npz is None:
        empty = np.full((40, 40, 3), 230, dtype=np.uint8)
        return empty, empty, "_No reachability artefact for this slice._"
    reach = npz["reach"]          # (MAX_DEPTH, n, n) uint8
    truth = npz["truth"]
    labels = npz["labels"]
    depth = int(np.clip(depth, 1, reach.shape[0]))
    pred = reach[depth - 1]
    f1 = _f1_offdiag(pred, truth)
    truth_img = _truth_image(truth, labels)
    pred_img = _pred_image(pred, truth, labels)
    note = (
        f"**diameter {diameter}**, **attention depth {depth}** → "
        f"same-component F1 = `{f1:.3f}`. "
        f"{'✅ closure recovered' if f1 >= 0.999 else '… still merging hops'}  \n"
        f"<small>predicted map colours: "
        f"<b style='color:#28a046'>green</b>=correct merge, "
        f"<b style='color:#d23232'>red</b>=false merge, "
        f"<b style='color:#f0961e'>orange</b>=missed (needs more hops). "
        f"Nodes reordered so each component is a contiguous block.</small>"
    )
    return truth_img, pred_img, note


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
def _refresh_run(run_name):
    abl = _load_ablation(run_name)
    df = _curve_df(abl)
    md = _headline_md(abl)
    diams = abl["diameters"] if abl else [3]
    canon = abl["canonical_diameter"] if abl else 3
    max_depth = max(abl["depths"]) if abl else 8
    diam_update = gr.update(choices=[int(d) for d in diams], value=int(canon))
    depth_update = gr.update(minimum=1, maximum=int(max_depth), value=int(canon), step=1)
    return df, md, diam_update, depth_update


with gr.Blocks(title="attention_connected_components · pass_2") as demo:
    gr.Markdown(
        "# attention_connected_components — pass_2\n"
        "A **hand-built attention circuit** (stacked adjacency-masked "
        "self-attention, identity values, no MLP) that recovers a graph's "
        "connected components. Depth = number of attention hops."
    )

    run_dd = gr.Dropdown(
        choices=_run_choices(),
        value=_latest_run_name(),
        label="results run",
        interactive=True,
    )

    with gr.Tab("Demo"):
        headline_md = gr.Markdown()

        gr.Markdown("### (A) Faithfulness: F1 vs attention depth")
        curve = gr.LinePlot(
            x="depth",
            y="f1",
            color="series",
            x_title="attention depth (number of hops)",
            y_title="same-component F1",
            title="Each hop adds one hop of reach; jumps to 1.0 at depth==diameter",
            height=340,
        )

        gr.Markdown("### (B) Reachability fills in hop-by-hop")
        with gr.Row():
            diam_dd = gr.Dropdown(choices=[1, 2, 3, 5], value=3,
                                  label="component diameter", interactive=True)
            depth_sl = gr.Slider(minimum=1, maximum=8, value=3, step=1,
                                 label="attention depth", interactive=True)
        with gr.Row():
            truth_im = gr.Image(label="ground truth (same component)",
                                type="numpy", height=300)
            pred_im = gr.Image(label="attention prediction vs truth",
                               type="numpy", height=300)
        hm_note = gr.Markdown()

        def _on_run(run_name):
            df, md, diam_u, depth_u = _refresh_run(run_name)
            return df, md, diam_u, depth_u

        def _on_heatmap(run_name, diameter, depth):
            return _render_heatmaps(run_name, diameter, depth)

        run_dd.change(
            _on_run,
            inputs=run_dd,
            outputs=[curve, headline_md, diam_dd, depth_sl],
        ).then(
            _on_heatmap,
            inputs=[run_dd, diam_dd, depth_sl],
            outputs=[truth_im, pred_im, hm_note],
        )
        diam_dd.change(_on_heatmap, inputs=[run_dd, diam_dd, depth_sl],
                       outputs=[truth_im, pred_im, hm_note])
        depth_sl.change(_on_heatmap, inputs=[run_dd, diam_dd, depth_sl],
                        outputs=[truth_im, pred_im, hm_note])

        def _init():
            run_name = _latest_run_name()
            df, md, diam_u, depth_u = _refresh_run(run_name)
            t, p, note = _render_heatmaps(run_name, 3, 3)
            return df, md, diam_u, depth_u, t, p, note

        demo.load(
            _init,
            inputs=None,
            outputs=[curve, headline_md, diam_dd, depth_sl,
                     truth_im, pred_im, hm_note],
        )

    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
