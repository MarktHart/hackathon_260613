"""Gradio app for attention_hierarchical_pool / pass_3 (hand_built).

Demo tab: shows the hand-built Gaussian attention head and the fine -> coarse
pooling shift with depth. Benchmark tab: cross-attempt leaderboard panel.

All demo-side compute is a NumPy mirror of main.py's GPU circuit (identical
math, no CUDA needed just to draw plots). The real GPU compute lives in main.py.
"""

import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from agentic.experiments import benchmark_panel

BASE = os.path.dirname(os.path.abspath(__file__))
GOAL_DIR = os.path.dirname(BASE)
RESULTS = os.path.join(BASE, "results")

# Constants mirroring main.py
SEQ_LEN = 256
CHUNK = 16
NLAY = 12
NHEAD = 8
SMIN = 0.55
SMAX = 7.0

# uniform-within-chunk strawman concentrations (from benchmark.py)
BASE_LOCAL = 74.0 / 256.0   # 0.289
BASE_CHUNK = 1.0
BASE_SPREAD = BASE_CHUNK / BASE_LOCAL  # ~3.46, but FLAT across depth -> robustness 1.0

_POS = np.arange(SEQ_LEN)
_SAME = (_POS[:, None] // CHUNK) == (_POS[None, :] // CHUNK)


def _sigma(layer: int, head: int) -> float:
    base = SMIN * (SMAX / SMIN) ** (layer / (NLAY - 1))
    return base * (0.7 + 0.6 * (head / (NHEAD - 1)))


def _attn(layer: int, head: int) -> np.ndarray:
    s = -0.5 * (_POS[:, None] - _POS[None, :]) ** 2 / _sigma(layer, head) ** 2
    s = np.where(_SAME, s, -1e30)
    s = s - s.max(axis=1, keepdims=True)
    e = np.where(_SAME, np.exp(s), 0.0)
    return e / e.sum(axis=1, keepdims=True)


def _conc(a: np.ndarray) -> tuple[float, float, float]:
    """local, chunk, entropy averaged over queries (matches task.py)."""
    loc = ch = ent = 0.0
    for q in range(SEQ_LEN):
        r = a[q]
        qc = q // CHUNK
        ch += r[qc * CHUNK: qc * CHUNK + CHUNK].sum()
        loc += sum(r[k] for k in range(max(0, q - 2), min(SEQ_LEN, q + 3)) if k // CHUNK == qc)
        rc = np.clip(r, 1e-12, 1.0)
        ent += -(rc * np.log(rc)).sum()
    return loc / SEQ_LEN, ch / SEQ_LEN, ent / SEQ_LEN


# ---------------------------------------------------------------- runs / json
def _runs() -> list[str]:
    if not os.path.isdir(RESULTS):
        return []
    out = []
    for d in sorted(os.listdir(RESULTS), reverse=True):
        if os.path.isfile(os.path.join(RESULTS, d, "benchmark.json")):
            out.append(d)
    return out


def _load_sweep(run_id: str | None) -> list[dict] | None:
    if not run_id:
        return None
    path = os.path.join(RESULTS, run_id, "benchmark.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("payload", data).get("sweep")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def _curve(sweep: list[dict] | None) -> dict[str, list[float]]:
    """Per-layer median local / chunk concentration and spread. Falls back to
    the analytic NumPy mirror when no recorded run is available."""
    layers = list(range(NLAY))
    local, chunk, spread = [], [], []
    if sweep:
        by_layer: dict[int, list[dict]] = {L: [] for L in layers}
        for rec in sweep:
            by_layer[rec["layer"]].append(rec)
        for L in layers:
            ls = [r["local_concentration"] for r in by_layer[L]]
            cs = [r["chunk_concentration"] for r in by_layer[L]]
            lm, cm = st.median(ls), st.median(cs)
            local.append(lm)
            chunk.append(cm)
            spread.append(cm / lm if lm > 0 else float("nan"))
    else:
        for L in layers:
            res = [_conc(_attn(L, h)) for h in range(NHEAD)]
            lm = st.median([x[0] for x in res])
            cm = st.median([x[1] for x in res])
            local.append(lm)
            chunk.append(cm)
            spread.append(cm / lm)
    return {"layer": layers, "local": local, "chunk": chunk, "spread": spread}


def _robustness(spread: list[float]) -> float:
    early = st.median(spread[: NLAY // 2])
    late = st.median(spread[NLAY // 2:])
    return late / early if early else float("nan")


# ---------------------------------------------------------------- figures
def _heatmap_fig(layer: int, head: int):
    a = _attn(layer, head)
    win = a[16:64, 16:64]  # 3 interior chunks (tokens 16..63), avoids seq edges
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(win, cmap="magma", aspect="equal", extent=[16, 64, 64, 16])
    for b in (32, 48):
        ax.axhline(b, color="cyan", lw=0.8, ls="--")
        ax.axvline(b, color="cyan", lw=0.8, ls="--")
    ax.set_title(f"Attention — layer {layer}, head {head}  (σ={_sigma(layer, head):.2f})")
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    fig.colorbar(im, ax=ax, fraction=0.046, label="attn weight")
    fig.tight_layout()
    return fig


def _depth_fig(sweep: list[dict] | None):
    c = _curve(sweep)
    L = c["layer"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    # left: the headline mechanism — within-chunk spread rising with depth
    ax1.plot(L, c["spread"], "o-", color="#d1495b", lw=2, label="this head circuit")
    ax1.axhline(BASE_SPREAD, color="gray", ls="--", lw=1.5,
                label="uniform-in-chunk (flat → robustness 1.0)")
    ax1.axvspan(-0.5, 5.5, color="#4a90d9", alpha=0.07)
    ax1.axvspan(5.5, 11.5, color="#d14949", alpha=0.07)
    ax1.set_xlabel("layer (early ◀ ─ ▶ late)")
    ax1.set_ylabel("spread = chunk_conc / local_conc")
    ax1.set_title(f"Fine→coarse pooling shift\nrobustness = {_robustness(c['spread']):.2f}  (>1 ⇒ hierarchy)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25)

    # right: the two concentrations that drive it
    ax2.plot(L, c["local"], "s-", color="#2e8b57", label="local conc (±2 window)")
    ax2.plot(L, c["chunk"], "^-", color="#8a5a00", label="chunk conc (16 tok)")
    ax2.axhline(BASE_LOCAL, color="#2e8b57", ls=":", lw=1, label="uniform local baseline")
    ax2.set_xlabel("layer")
    ax2.set_ylabel("attention mass")
    ax2.set_title("Local mass collapses with depth\n(chunk mass stays ≈1: pooling respects chunks)")
    ax2.legend(fontsize=8, loc="center left")
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _stats_md(layer: int, head: int) -> str:
    loc, ch, ent = _conc(_attn(layer, head))
    return (
        f"**Layer {layer}, head {head}** — σ = `{_sigma(layer, head):.2f}`\n\n"
        f"| local conc | chunk conc | spread | entropy (nats) |\n"
        f"|---|---|---|---|\n"
        f"| {loc:.3f} | {ch:.3f} | {ch / loc:.3f} | {ent:.3f} |\n\n"
        f"Early layers (small σ) concentrate mass on the query token → high local "
        f"conc, spread ≈ 1. Late layers (large σ) spread mass across the whole "
        f"chunk → low local conc, spread ≫ 1. That rising spread *is* hierarchical pooling."
    )


# ---------------------------------------------------------------- app
with gr.Blocks() as demo:
    gr.Markdown("# attention_hierarchical_pool — pass_3 (hand_built)")
    gr.Markdown(
        "A single attention head whose **hand-set Q/K weights** realise a Gaussian "
        "distance kernel `exp(-(i-j)²/2σ²)` masked to the query's chunk. The width "
        "**σ is the only thing that grows with depth** — early layers pool a tight "
        "local window, late layers pool the full 16-token chunk. The benchmark "
        "rewards exactly this fine→coarse shift."
    )

    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                with gr.Column(scale=1):
                    layer_sl = gr.Slider(0, NLAY - 1, value=0, step=1, label="layer")
                    head_sl = gr.Slider(0, NHEAD - 1, value=3, step=1, label="head")
                    heat = gr.Plot(label="Per-head attention (3 interior chunks)")
                    stats = gr.Markdown()
                with gr.Column(scale=1):
                    run_dd = gr.Dropdown(
                        choices=_runs(), value=(_runs()[0] if _runs() else None),
                        label="benchmark run (latest by default)",
                    )
                    depth = gr.Plot(label="Depth curves (median over heads)")

            def _on_head(layer, head):
                return _heatmap_fig(int(layer), int(head)), _stats_md(int(layer), int(head))

            def _on_run(run_id):
                return _depth_fig(_load_sweep(run_id))

            layer_sl.change(_on_head, [layer_sl, head_sl], [heat, stats])
            head_sl.change(_on_head, [layer_sl, head_sl], [heat, stats])
            run_dd.change(_on_run, run_dd, depth)

            demo.load(_on_head, [layer_sl, head_sl], [heat, stats])
            demo.load(_on_run, run_dd, depth)

        with gr.Tab("Benchmark"):
            benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
