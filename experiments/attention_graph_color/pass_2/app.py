import gradio as gr
import json
from pathlib import Path

from agentic.experiments import load_task, benchmark_panel, results_dir


# Resolve the goal directory (parent of this attempt's folder)
GOAL_DIR = Path(__file__).parent.parent


def _load_latest_payload():
    """Load the most recent benchmark.json from this attempt's results/ directory."""
    run_dir = results_dir(__file__)
    # results_dir returns the latest run directory; benchmark.json is inside it
    payload_path = run_dir / "benchmark.json"
    if not payload_path.exists():
        return {"error": f"No benchmark.json found at {payload_path}"}
    try:
        with payload_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


def _format_metrics(payload: dict) -> str:
    """Format key metrics from payload for the Demo tab Markdown."""
    if "error" in payload:
        return f"**Error loading payload:** {payload['error']}"

    # Import benchmark to recompute metrics exactly as the dashboard does
    from experiments.attention_graph_color import benchmark as bench_module
    try:
        m = bench_module.score(payload)
    except Exception as e:
        return f"**Error computing metrics:** {e}"

    sep_canon = m.get("color_separation_canonical", 0.0)
    edge_respect = m.get("edge_respect_canonical", 0.0)
    lift = m.get("lift_over_linear_baseline", 0.0)
    sep_overall = m.get("color_separation_overall", 0.0)
    invalid = m.get("invalid_edge_attention_canonical", 0.0)
    baseline = m.get("linear_baseline_color_separation", 0.0)

    return f"""
**Statistics for the canonical n=40 slice** (all p, 15 graphs):

- **`color_separation_canonical`**: {sep_canon:.6f}
- **`edge_respect_canonical`**: {edge_respect:.6f}
- **`lift_over_linear_baseline`**: {lift:.6f}
- **`color_separation_overall`**: {sep_overall:.6f} (all 45 graphs)
- **`invalid_edge_attention_canonical`**: {invalid:.6f} (sanity invariant, should be 0)
- **`linear_baseline_color_separation`**: {baseline:.6f} (uniform attention reference)

The hand-built mechanism uses a fixed colour-difference projector so that
every edge (which connects different colours by construction) receives
positive attention while all non-edges and same-colour pairs receive zero.
This yields maximal separation and edge-respect scores.
"""


with gr.Blocks() as demo:
    # ---- Demo Tab ----
    gr.Markdown("# Proper Graph Coloring Attention — Demo")
    gr.Markdown(
        """
        This demo visualises a **hand-built attention mechanism** that respects the structure of a proper graph coloring.
        No learning is involved — the circuit is hand-coded with fixed weights on the GPU.
        
        The mechanism:
        1. Takes the one-hot colour features (first k columns of `feats`).
        2. Applies a fixed projector P where P[i,j] = 1 if i ≠ j else 0.
        3. Computes Q = colours @ P, K = colours, giving S = QKᵀ with S_ij = 1 iff colours differ.
        4. Masks with adjacency: only edges (always cross-colour in a proper coloring) get mass.
        5. Row-normalises; isolated nodes get zero rows.
        
        Because the projector explicitly encodes colour difference and the adjacency mask
        selects exactly the edges, the attention matrix places **all mass on cross-colour edges**
        and zero on same-colour pairs and non-edges.
        """
    )

    summary_md = gr.Markdown("Loading latest run…")

    def _refresh_summary():
        payload = _load_latest_payload()
        return _format_metrics(payload)

    # Load on startup
    demo.load(_refresh_summary, inputs=None, outputs=summary_md, queue=False)

    # Manual refresh button
    refresh_btn = gr.Button("Refresh from latest run")
    refresh_btn.click(_refresh_summary, inputs=None, outputs=summary_md, queue=False)

    gr.Markdown("---")

    # ---- Benchmark Tab ----
    gr.Markdown("# Benchmark Dashboard")
    gr.Markdown(
        """
        The dashboard below scans every attempt under this goal, shows a leaderboard,
        and plots metric history over runs. It is the canonical cross-attempt view
        provided by `agentic.experiments.benchmark_panel`.
        """
    )
    panel = benchmark_panel(str(GOAL_DIR))
    if panel is not None:
        panel.render()
    else:
        gr.Markdown("_(benchmark panel unavailable — run an attempt first)_")


if __name__ == "__main__":
    demo.launch()