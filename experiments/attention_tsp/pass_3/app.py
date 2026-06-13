"""
attention_tsp / pass_3 — Gradio app.

Demo tab tells the mechanistic story in two views:
  (A) "Attention = -distance": for one instance and a chosen current city,
      colour every city by its raw attention score Q.K and mark the argmax.
      A scatter of (score) vs (-true squared distance) collapses onto y=x,
      visually proving the QK head computes negative squared distance.
  (B) "What feature does the work?": a grouped bar chart of step-wise NN
      accuracy per problem size for the full mechanism, the two ablations,
      and the random baseline (read from the latest run's ablation.json).

Benchmark tab: agentic.experiments.benchmark_panel across all attempts.
"""

import json
from pathlib import Path

import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel, results_dir, load_task

task = load_task(__file__)

# --- NumPy mirror of the hand-set attention (CPU is fine for plotting) -------
W_Q_FULL = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
W_K_FULL = np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, -1], [0, 0, -1, 0]], float)


def phi(coords):
    x, y = coords[:, 0], coords[:, 1]
    s = x * x + y * y
    return np.stack([x, y, s, np.ones_like(x)], axis=1)


def attention_scores(coords, current_idx, mode="full"):
    Wq, Wk = W_Q_FULL.copy(), W_K_FULL.copy()
    if mode == "ablate_key_norm":
        Wk[3, :] = 0.0
    elif mode == "ablate_query_norm":
        Wq[2, :] = 0.0
    P = phi(coords)
    Q = P @ Wq.T
    K = P @ Wk.T
    return K @ Q[current_idx]


def _instance(n_cities, instance_idx):
    batch = task.generate(task.EVAL_SEED)
    idxs = [i for i, n in enumerate(batch.ns) if n == int(n_cities)]
    if not idxs:
        idxs = list(range(len(batch.ns)))
    pick = idxs[int(instance_idx) % len(idxs)]
    return batch.coords_list[pick]


# --- View A: attention lands on the nearest city -----------------------------
def view_attention(n_cities, instance_idx, current_idx, mode):
    coords = _instance(n_cities, instance_idx)
    n = coords.shape[0]
    current_idx = int(current_idx) % n
    scores = attention_scores(coords, current_idx, mode)

    # mask the current city itself for the "next city" argmax
    cand = scores.copy()
    cand[current_idx] = -np.inf
    choice = int(np.argmax(cand))

    # true nearest city (ground truth)
    d2 = ((coords - coords[current_idx]) ** 2).sum(axis=1)
    d2_masked = d2.copy()
    d2_masked[current_idx] = np.inf
    true_nn = int(np.argmin(d2_masked))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left: cities coloured by attention score; lines to chosen + true NN
    sc = axL.scatter(coords[:, 0], coords[:, 1], c=scores, cmap="viridis",
                     s=160, zorder=3, edgecolors="k", linewidths=0.4)
    axL.scatter(*coords[current_idx], c="white", s=230, marker="o",
                edgecolors="red", linewidths=2.5, zorder=4, label="current city")
    axL.plot([coords[current_idx, 0], coords[choice, 0]],
             [coords[current_idx, 1], coords[choice, 1]],
             "r-", lw=2.5, zorder=2, label=f"attention argmax → city {choice}")
    if true_nn != choice:
        axL.plot([coords[current_idx, 0], coords[true_nn, 0]],
                 [coords[current_idx, 1], coords[true_nn, 1]],
                 "k--", lw=1.5, zorder=1, label=f"true nearest → city {true_nn}")
    for i, (x, y) in enumerate(coords):
        axL.annotate(str(i), (x, y), xytext=(4, 4),
                     textcoords="offset points", fontsize=7)
    axL.set_title(f"Attention scores  (mode={mode})\nargmax {'==' if choice==true_nn else '!='} true nearest")
    axL.set_xlim(-0.05, 1.05); axL.set_ylim(-0.05, 1.05); axL.set_aspect("equal")
    axL.legend(fontsize=7, loc="upper left")
    fig.colorbar(sc, ax=axL, fraction=0.046, label="Q·K score")

    # Right: score vs -squared distance — proves QK == -||q-k||^2 (full mode)
    axR.scatter(-d2, scores, c="tab:blue", s=40, zorder=3)
    lo = float(min((-d2).min(), scores.min()))
    hi = float(max((-d2).max(), scores.max()))
    axR.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x")
    axR.set_xlabel("-‖current − city‖²  (true neg. sq. distance)")
    axR.set_ylabel("attention score  Q·K")
    if mode == "full":
        axR.set_title("Full head: score lies exactly on y = x")
    else:
        axR.set_title(f"{mode}: score departs from -distance")
    axR.legend(fontsize=8)
    axR.grid(alpha=0.3)

    fig.tight_layout()
    verdict = (
        f"Current city {current_idx}: attention argmax = city {choice}; "
        f"true nearest = city {true_nn}.  "
        + ("✅ MATCH" if choice == true_nn else "❌ MISMATCH (mechanism broken by ablation)")
    )
    return fig, verdict


# --- View B: faithfulness / ablation bar chart -------------------------------
def _latest_ablation():
    base = results_dir(__file__).parent
    if not base.exists():
        return None
    runs = sorted([d for d in base.iterdir() if d.is_dir()])
    for d in reversed(runs):
        p = d / "ablation.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def view_ablation():
    ab = _latest_ablation()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if ab is None:
        ax.text(0.5, 0.5, "No ablation.json yet — run main.py first.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    ns = ab["n_cities_sweep"]
    series = {
        "full (Q·K = -dist²)": ([r["nn_accuracy"] for r in ab["variants"]["full"]], "tab:green"),
        "ablate key-norm -s_j": ([r["nn_accuracy"] for r in ab["variants"]["ablate_key_norm"]], "tab:red"),
        "ablate query-norm s_i": ([r["nn_accuracy"] for r in ab["variants"]["ablate_query_norm"]], "tab:orange"),
        "random baseline": ([r["nn_accuracy"] for r in ab["random_baseline"]], "gray"),
    }
    x = np.arange(len(ns))
    w = 0.2
    for k, (label, (vals, color)) in enumerate(series.items()):
        ax.bar(x + (k - 1.5) * w, vals, w, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in ns])
    ax.set_ylabel("step-wise NN accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Faithfulness: which feature does the work?\n"
                 "key-norm ablation collapses to ~random; query-norm ablation is inert")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


MECH_MD = """
## The QK head *is* nearest-neighbour

This is `base_model.py`'s single-head `Attention` with **two frozen changes**:
the learned token embedding becomes a fixed quadratic feature map
`φ(x,y) = [x, y, x²+y², 1]`, and `W_q`, `W_k` are hand-set 4×4 matrices.

With `s = x²+y²`:

```
Q_i = φ_i               = [x_i, y_i, s_i, 1]
K_j = W_k φ_j           = [2x_j, 2y_j, -1, -s_j]
Q_i · K_j = 2(x_i x_j + y_i y_j) - s_i - s_j = -‖c_i - c_j‖²
```

So the ordinary dot-product attention score **equals the negative squared
distance** — argmax over unvisited cities is the nearest one. No bespoke
distance call; the heuristic falls out of standard attention once the key
carries a `-‖k‖²` feature.
"""


with gr.Blocks(title="attention_tsp · pass_3 · QK = -distance") as demo:
    with gr.Tab("Demo: attention = nearest-neighbour"):
        gr.Markdown(MECH_MD)
        with gr.Row():
            n_in = gr.Slider(5, 40, value=10, step=5, label="Problem size N")
            inst_in = gr.Number(value=0, precision=0, label="Instance index")
            cur_in = gr.Number(value=0, precision=0, label="Current city index")
            mode_in = gr.Radio(
                ["full", "ablate_key_norm", "ablate_query_norm"],
                value="full", label="Mechanism mode",
            )
        go = gr.Button("Show attention", variant="primary")
        att_plot = gr.Plot(label="Attention scores & score-vs-distance")
        verdict = gr.Markdown()
        go.click(view_attention, [n_in, inst_in, cur_in, mode_in],
                 [att_plot, verdict])

        gr.Markdown("---\n### Causal evidence (latest run)")
        refresh = gr.Button("Refresh ablation chart")
        ab_plot = gr.Plot(label="NN accuracy by variant × problem size")
        refresh.click(view_ablation, outputs=ab_plot)

        demo.load(view_attention, [n_in, inst_in, cur_in, mode_in],
                  [att_plot, verdict])
        demo.load(view_ablation, outputs=ab_plot)

    with gr.Tab("Benchmark: all attempts"):
        gr.Markdown(
            "Leaderboard across every `attention_tsp` attempt. Headline metric is "
            "`size_robustness` (NN accuracy retained from N=5 to N=40)."
        )
        benchmark_panel("..")


if __name__ == "__main__":
    demo.launch()
