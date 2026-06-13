
import gradio as gr

from agentic.experiments import benchmark_panel, load_task


# ---------------------------------------------------------------------------
# This file builds a Gradio demo with two tabs:
#   1. Demo tab: a small interactive view and a per-graph summary table.
#   2. Benchmark tab: the canonical dashboard that aggregates across all attempts.
#
# The pipeline boot-check imports this module to assert `demo` is a `gr.Blocks`,
# and that all event handlers (.click, .load, etc.) live inside the `with gr.Blocks()` block.
# ---------------------------------------------------------------------------


with gr.Blocks() as demo:
    # Demo Tab
    gr.Markdown("# Proper Graph Coloring Attention Demo")
    gr.Markdown(
        """
        This demo visualises a **hand-built attention mechanism** that respects the structure of a proper graph coloring. No machine learning is involved — the circuit is hand-coded. It shows how an attention matrix can be constructed to:
        
        - Place mass predominantly on **edges** (where `adj[i,j]==1`).
        - Within those edges, give **significantly larger attention to differently coloured node pairs** than to same-coloured pairs — a constraint that is automatically satisfied because the coloring algorithm guarantees no edge connects two nodes of the same colour (i.e., `cross_edge_same_color` is identically zero as a sanity invariant).
        
        Below are the statistics for the canonical (n=40, p=0.2) slice of the sweep, plus the overall baseline comparison. The key headline is `color_separation_canonical`, which measures the excess attention on differently coloured pairs over same-coloured pairs. A positive value indicates the coloring distinction is encoded in the mechanism; a value close to zero would mean the attention is effectively uniform (the structureless baseline).
        """
    )

    # Show the most recent payload summary. This is populated by a demo load hook.
    with gr.Blocks() as stats_panel:
        summary_text = gr.Markdown("""
        **Statistics for the canonical n=40 slice** (p=0.2):

        - **`color_separation_canonical`**: {_color_sep_}

        - **`edge_respect_canonical`**: {_edge_respect_}

        - **`lift_over_linear_baseline`**: {_lift_}

        - **`color_separation_overall`**: {_overall_sep_}

        Baseline (uniform attention) gives zero separation, so any positive lift confirms the coloring circuit is active across the batch. The dashboard on the Benchmark tab groups these metrics across all attempts at the same goal, letting you compare multiple strategies at once.
        """)


    def _load_payload(goal_dir, run_dir):
        """
        Load the payload from `run_dir` and format its headline metrics as a dict of strings for the Markdown panel above.
        """
        try:
            import json
            payload_path = run_dir.joinpath("benchmark.json")
            with payload_path.open encoding="utf-8") as f:
                payload = json.load(f)
            m = benchmark.score(payload)  # recompute so metrics are exactly as the dashboard sees them
            return {
                "_color_sep_": f"{m['color_separation_canonical']:.4f}",
                "_edge_respect_": f"{m['edge_respect_canonical']:.4f}",
                "_lift_": f"{m['lift_over_linear_baseline']:.4f}",
                "_overall_sep_": f"{m['color_separation_overall']:.4f}",
            }
        except Exception as e:
            return {"error": str(e)}  # shows an error block instead of breaking the UI

    demo.load(
        _load_payload,  # runs on first page load; updates the Markdown panel with the latest payload stats
        inputs=[gr.JSON(value=load_task(__file__).__name__)] + [gr.Text(value=run_dir)],
        outputs=[stats_panel],
        queue=False,
    )

    # Benchmark Tab: drop in the canonical leaderboard / history dashboard
    with gr.Blocks()
        # Demo Tab
        gr.Markdown("# Proper Graph Coloring Attention Demo")
        gr.Markdown(
            """
            This demo visualises a **hand-built attention mechanism** that respects the structure of a proper graph coloring. No machine learning is involved — the circuit is hand-coded. It shows how an attention matrix can be constructed to:
            
            - Place mass predominantly on **edges** (where `adj[i,j]==1`).
            - Within those edges, give **significantly larger attention to differently coloured node pairs** than to same-coloured pairs — a constraint that is automatically satisfied because the coloring algorithm guarantees no edge connects two nodes of the same colour (i.e., `cross_edge_same_color` is identically zero as a sanity invariant).
            
            Below are the statistics for the canonical (n=40, p=0.2) slice of the sweep, plus the overall baseline comparison. The key headline is `color_separation_canonical`, which measures the excess attention on differently coloured pairs over same-coloured pairs. A positive value indicates the coloring distinction is encoded in the mechanism; a value close to zero would mean the attention is effectively uniform (the structureless baseline).
            """
        )

        # Show the most recent payload summary. This is populated by a demo load hook.
        with gr.Blocks() as stats_panel:
            summary_text = gr.Markdown("""
            **Statistics for the canonical n=40 slice** (p=0.2):

            - **`color_separation_canonical`**: {_color_sep_}

            - **`edge_respect_canonical`**: {_edge_respect_}

            - **`lift_over_linear_baseline`**: {_lift_}

            - **`color_separation_overall`**: {_overall_sep_}

            Baseline (uniform attention) gives zero separation, so any positive lift confirms the coloring circuit is active across the batch. The dashboard on the Benchmark tab groups these metrics across all attempts at the same goal, letting you compare multiple strategies at once.
            """)


        def _load_payload(goal_dir, run_dir):
            """
            Load the payload from `run_dir` and format its headline metrics as a dict of strings for the Markdown panel above.
            """
            try:
                import json
                payload_path = run_dir.joinpath("benchmark.json")
                with payload_path.open encoding="utf-8") as f:
                    payload = json.load(f)
                m = benchmark.score(payload)  # recompute so metrics are exactly as the dashboard sees them
                return {
                    "_color_sep_": f"{m['color_separation_canonical']:.4f}",
                    "_edge_respect_": f"{m['edge_respect_canonical']:.4f}",
                    "_lift_": f"{m['lift_over_linear_baseline']:.4f}",
                    "_overall_sep_": f"{m['color_separation_overall']:.4f}",
                }
            except Exception as e:
                return {"error": str(e)}  # shows an error block instead of breaking the UI

        demo.load(
            _load_payload,  # runs on first page load; updates the Markdown panel with the latest payload stats
            inputs=[gr.JSON(value=load_task(__file__).__name__)] + [gr.Text(value=run_dir)],
            outputs=[stats_panel],
            queue=False,
        )

        # Benchmark Tab: drop in the canonical leaderboard / history dashboard
        with gr.Blocks():
            gr.Markdown("# Benchmark Dashboard")
            gr.Markdown(
                """
                The gradio app runs the demo tab above on the latest run by default. The benchmark tab shows a cross-attempt leaderboard, plots metric history over runs, and lets you compare different strategies on the same task. It sources every attempt's `benchmark.json` from the same goal directory, with automatic version filtering (only the highest version present is shown). The dashboard is built from `agentic.experiments.benchmark_panel` and requires no extra code — just drop it in as below. The pipeline pre-renders the dashboard at import time for the boot-check; it does not need to launch the demo or bind a port for that check, though it launches the demo when you open the UI in the notebook.
                """
            )
            panel = benchmark_panel(goal_dir)
            panel.render()


if __name__ == "__main__":
    demo.launch()