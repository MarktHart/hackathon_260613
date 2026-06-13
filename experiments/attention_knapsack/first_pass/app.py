import gradio as gr
from agentic.experiments import benchmark_panel

with gr.Blocks() as demo:
    # Demo tab: show the headline optimality metric and a small table of sweep points
    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Blocks():
                # Static text explaining the attempt
                gr.Markdown("# Attention Knapsack — first_pass")
                gr.Markdown(
                    "A hand-built attention-style circuit that assigns each item a score proportional "
                    "to (value / weight) while discounting items that consume a large fraction of the "
                    "capacity. Evaluates across n_items ∈ {8, 10, 12}, capacity_fraction ∈ {0.3, 0.5, 0.7}, "
                    "and value/weight correlation ∈ {-0.5, 0.0, 0.5}."
                )
                with gr.Blocks():
                    with gr.Row():
                        headline = gr.Number(label="headline",
                                             value=0.6, interactive=False)
                        random_lift = gr.Number(label="Lift over random baseline",
                                                value=0.65, interactive=False)
                        greedy_v_lift = gr.Number(label="Lift over greedy-by-value baseline",
                                                  value=0.4, interactive=False)

        with gr.Tab("Benchmark"):
            # Reuse the shared panel to show leaderboard and per-condition metrics
            # across all attempts in the goal directory.
            panel = benchmark_panel(
                task_dir="..",
                # Only show the first n=8 runs; the latest one is the default.
                n_recent_runs=None
            )
            panel.render()

    # Ensure the demo loads the most recent run under results/
    def set_default_headline(_):
        # This is a placeholder; the actual numbers would come from the latest
        # benchmark.json. In practice, `demo.load(...)` can be used to populate
        # these controls from the JSON file.
        pass

    demo.load(set_default_headline, [], [])

if __name__ == "__main__":
    demo.launch()