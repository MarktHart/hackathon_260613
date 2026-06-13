import gradio as gr
import numpy as np
import json
from pathlib import Path

from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent


def _get_latest_run_dir() -> Path:
    results_root = Path(__file__).parent / "results"
    if not results_root.exists():
        return None
    run_dirs = sorted(
        [d for d in results_root.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return run_dirs[0] if run_dirs else None


def _load_latest_payload():
    run_dir = _get_latest_run_dir()
    if run_dir is None:
        return None, None
    benchmark_path = run_dir / "benchmark.json"
    if not benchmark_path.exists():
        return None, None
    with open(benchmark_path) as f:
        payload = json.load(f)
    return payload, run_dir


def _load_canonical_artefacts(run_dir: Path):
    attn_path = run_dir / "canonical_attention.npy"
    adj_path = run_dir / "canonical_adjacency.npy"
    if not attn_path.exists() or not adj_path.exists():
        return None, None
    return np.load(attn_path), np.load(adj_path)


def _compute_ancestors(adj: np.ndarray) -> np.ndarray:
    """Transitive closure (boolean)."""
    reach = adj.astype(bool).copy()
    n = reach.shape[0]
    for k in range(n):
        reach |= reach[:, k:k+1] & reach[k:k+1, :]
    return reach


def _topo_respect_single(attn: np.ndarray, anc: np.ndarray) -> float:
    """Fraction of ordered ancestor pairs respected."""
    a_idx, d_idx = np.where(anc)
    if a_idx.size == 0:
        return 0.5
    back = attn[d_idx, a_idx]
    fwd = attn[a_idx, d_idx]
    credit = np.where(back > fwd, 1.0, np.where(back == fwd, 0.5, 0.0))
    return float(credit.mean())


def _make_heatmap_markdown(attn: np.ndarray, adj: np.ndarray, dag_idx: int, head_idx: int = 0) -> str:
    """Textual heatmap with ancestor highlights."""
    n = attn.shape[0]
    anc = _compute_ancestors(adj)

    # Row/col labels
    labels = [f"{i}" for i in range(n)]

    lines = []
    lines.append(f"**DAG {dag_idx} — Attention Heatmap** (row=query, col=key)")
    lines.append(f"*Ancestor→descendant pairs highlighted: query (descendant) attends to key (ancestor)*")
    lines.append("")

    # Header
    header = "       " + "  ".join(f"{lab:>5}" for lab in labels)
    lines.append(header)

    for q in range(n):
        row_vals = []
        for k in range(n):
            val = attn[q, k]
            is_anc = anc[k, q]  # k is ancestor of q
            marker = " ★" if is_anc else ""
            row_vals.append(f"{val:5.2f}{marker}")
        lines.append(f"  {q:2d}:  " + "  ".join(row_vals))

    lines.append("")
    lines.append("★ = key is ancestor of query (should have higher attention)")
    return "\n".join(lines)


def _make_summary_markdown(payload: dict, run_dir: Path, dag_idx: int = 0) -> str:
    attn, adj = _load_canonical_artefacts(run_dir)
    if attn is None:
        return "No canonical artefacts found."

    this_attn = attn[dag_idx]
    this_adj = adj[dag_idx]
    anc = _compute_ancestors(this_adj)

    respect = _topo_respect_single(this_attn, anc)

    # Per-density summary from payload
    sweep = payload.get("sweep", [])
    density_lines = []
    for rec in sweep:
        d = rec["density"]
        tr = rec["topo_respect"]
        lift = rec["topo_respect"] - rec["uniform_respect"]
        density_lines.append(f"  - Density {d:.1f}: topo_respect={tr:.3f}, lift={lift:+.3f}")

    canonical = payload.get("canonical_density", 0.3)
    lines = [
        f"## Run Summary (canonical density = {canonical})",
        f"**Selected DAG {dag_idx} topo_respect: {respect:.3f}**",
        "",
        "**Sweep results:**",
    ] + density_lines + [
        "",
        f"**Model:** {payload.get('model_name', 'unknown')}",
        f"**Nodes:** {payload.get('n_nodes', '?')}, **DAGs/density:** {payload.get('n_dags', '?')}",
    ]
    return "\n".join(lines)


with gr.Blocks(title="Attention Topo Sort — Pass 2") as demo:
    gr.Markdown("# Attention Topo Sort — Pass 2")
    gr.Markdown(
        "Hand-built attention circuit: transitive closure on GPU. "
        "Each node attends to its ancestors in the DAG. "
        "Demo shows per-DAG heatmaps and sweep metrics."
    )

    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                with gr.Column(scale=1):
                    run_selector = gr.Dropdown(
                        choices=["(latest)"],
                        value="(latest)",
                        label="Run",
                        interactive=False
                    )
                    dag_selector = gr.Slider(
                        minimum=0, maximum=23, step=1, value=0,
                        label="DAG index (canonical density = 0.3)"
                    )
                    refresh_btn = gr.Button("Refresh from disk")

                    summary_md = gr.Markdown()

                with gr.Column(scale=2):
                    heatmap_md = gr.Markdown()

            def _on_load():
                payload, run_dir = _load_latest_payload()
                if payload is None:
                    return "(no runs yet)", "No runs found. Run `main.py` first."
                run_name = run_dir.name if run_dir else "unknown"
                summary = _make_summary_markdown(payload, run_dir, 0)
                attn, adj = _load_canonical_artefacts(run_dir)
                if attn is None:
                    heatmap = "Canonical artefacts not found. Re-run main.py."
                else:
                    heatmap = _make_heatmap_markdown(attn[0], adj[0], 0)
                return summary, heatmap

            def _on_dag_change(dag_idx, payload_state, run_dir_state):
                if payload_state is None or run_dir_state is None:
                    return "No data", "No data"
                summary = _make_summary_markdown(payload_state, Path(run_dir_state), int(dag_idx))
                attn, adj = _load_canonical_artefacts(Path(run_dir_state))
                if attn is None:
                    heatmap = "Artefacts missing"
                else:
                    heatmap = _make_heatmap_markdown(attn[int(dag_idx)], adj[int(dag_idx)], int(dag_idx))
                return summary, heatmap

            # State to hold loaded payload/run_dir
            payload_state = gr.State(value=None)
            run_dir_state = gr.State(value=None)

            def _load_and_store():
                payload, run_dir = _load_latest_payload()
                if payload is None:
                    return "(no runs)", "No runs", None, None
                run_name = run_dir.name if run_dir else "unknown"
                summary = _make_summary_markdown(payload, run_dir, 0)
                attn, adj = _load_canonical_artefacts(run_dir)
                if attn is None:
                    heatmap = "Artefacts missing"
                else:
                    heatmap = _make_heatmap_markdown(attn[0], adj[0], 0)
                return summary, heatmap, payload, str(run_dir) if run_dir else None

            demo.load(
                _load_and_store,
                outputs=[summary_md, heatmap_md, payload_state, run_dir_state]
            )

            refresh_btn.click(
                _load_and_store,
                outputs=[summary_md, heatmap_md, payload_state, run_dir_state]
            )

            dag_selector.change(
                _on_dag_change,
                inputs=[dag_selector, payload_state, run_dir_state],
                outputs=[summary_md, heatmap_md]
            )

        with gr.Tab("Benchmark"):
            gr.Markdown("## Benchmark History Across All Attempts")
            benchmark_panel(str(GOAL_DIR))

if __name__ == "__main__":
    demo.launch()