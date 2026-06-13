import json
from pathlib import Path

import gradio as gr
import pandas as pd

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = Path(__file__).parent
GOAL_DIR = ATTEMPT_DIR.parent
RESULTS = ATTEMPT_DIR / "results"


def list_runs():
    if not RESULTS.exists():
        return []
    return sorted([p.name for p in RESULTS.iterdir() if p.is_dir()], reverse=True)


def _load(run):
    if not run:
        return None
    f = RESULTS / run / "artifacts.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


EMPTY_PH = pd.DataFrame({"head": [], "copy_fidelity": []})
EMPTY_ABL = pd.DataFrame({"token": [], "copy_fidelity": [], "condition": []})
EMPTY_DIAG = pd.DataFrame({"head": [], "diagonal_mass": []})


def refresh(run):
    art = _load(run)
    if art is None:
        return (EMPTY_PH, EMPTY_ABL, EMPTY_DIAG, pd.DataFrame(),
                "**No runs yet.** Run `main.py` to generate results.")

    ph = art["per_head"]
    heads = [f"h{i}" for i in ph["head"]]
    ph_df = pd.DataFrame({"head": heads, "copy_fidelity": ph["fidelity"]})
    diag_df = pd.DataFrame({"head": heads, "diagonal_mass": ph["diag_mass"]})

    rows = []
    for r in art["sweep_real"]:
        rows.append({"token": str(r["token"]), "copy_fidelity": r["fidelity"],
                     "condition": "identity head (best)"})
    for r in art["sweep_ablated"]:
        rows.append({"token": str(r["token"]), "copy_fidelity": r["fidelity"],
                     "condition": "ablated (no position)"})
    abl_df = pd.DataFrame(rows)

    attn = art["head0_attn"]
    attn_df = pd.DataFrame(
        [[round(v, 3) for v in row] for row in attn],
        columns=[f"k{j}" for j in range(len(attn))],
    )
    attn_df.insert(0, "query", [f"q{i}" for i in range(len(attn))])

    ct = art["canonical_token"]
    can = next(r for r in art["sweep_real"] if r["token"] == ct)
    abl_can = next(r for r in art["sweep_ablated"] if r["token"] == ct)
    md = (
        f"### Identity-copy head — canonical token **{ct}**\n"
        f"- Best head: **h{can['best_head']}**  |  copy fidelity: **{can['fidelity']:.4f}**  "
        f"|  diagonal mass: **{can['diag']:.4f}**\n"
        f"- Uniform-attention baseline ≈ **0.25**  →  lift **{can['fidelity'] - 0.25:+.4f}**\n"
        f"- **Causal ablation** (zero the positional subspace): fidelity collapses to "
        f"**{abl_can['fidelity']:.4f}** — the diagonal copy is *driven by* the positional subspace."
    )
    return ph_df, abl_df, diag_df, attn_df, md


with gr.Blocks(title="Attention Identity Copy — pass_5") as demo:
    gr.Markdown(
        "# Attention Identity Copy — hand-built head\n"
        "A single real attention layer (no MLP). Scores come from `softmax(q·kᵀ)` where "
        "`q,k` read only a **one-hot positional subspace** of the residual stream, so the "
        "softmax concentrates on the diagonal (i→i) and the head copies the value at the "
        "same position. Because the sweep makes every position the *same* token, only the "
        "positional subspace can produce a diagonal — which the ablation exploits."
    )

    runs = list_runs()
    with gr.Tabs():
        with gr.Tab("Demo"):
            run_dd = gr.Dropdown(
                choices=runs, value=(runs[0] if runs else None),
                label="Run (latest first)",
            )
            md_box = gr.Markdown()
            with gr.Row():
                ph_plot = gr.BarPlot(
                    x="head", y="copy_fidelity", y_lim=[0, 1],
                    title="Per-head copy fidelity @ token 128 (h0 = identity copier, h7 = uniform strawman)",
                )
                diag_plot = gr.BarPlot(
                    x="head", y="diagonal_mass", y_lim=[0, 1],
                    title="Diagonal attention mass per head",
                )
            abl_plot = gr.BarPlot(
                x="token", y="copy_fidelity", color="condition", y_lim=[0, 1],
                title="Causal ablation: identity head vs. positional subspace zeroed (across the vocab sweep)",
            )
            gr.Markdown(
                "Head-0 attention matrix at the canonical token (rows=query, cols=key). "
                "Mass on the diagonal == identity copy:"
            )
            attn_tbl = gr.Dataframe(interactive=False, wrap=True)

            outs = [ph_plot, abl_plot, diag_plot, attn_tbl, md_box]
            run_dd.change(refresh, inputs=run_dd, outputs=outs)
            demo.load(refresh, inputs=run_dd, outputs=outs)

        with gr.Tab("Benchmark"):
            benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
