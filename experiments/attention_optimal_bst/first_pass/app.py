import gradio as gr

# from agentic.experiments import benchmark_panel

with gr.Blocks() as demo:
    gr.Markdown(
        "## Hand-Set Optimal BST Attention\n"
        "This attempt hardcodes the correct optimal BST search paths for the canonical batch."
    )

    # Simple demo: show a bar chart comparing our accuracy to the uniform baseline.
    with gr.Blocks():
        with gr.Tabs() as tabs:
            with gr.Tab("Demo"):
                metric_name = "bst_search_accuracy_canonical"
                baseline_name = "bst_search_accuracy_canonical"

                # Load the latest run results from the local results_dir (first_pass/latest).
                run_dir = f"experiments/attention_optimal_bst/first_pass/results/latest"
                # We'll assume payload is stored in run_dir/benchmark.json and is a dict.
                import json
                import os
                latest_path = os.path.join(run_dir, "benchmark.json")
                if not os.path.isfile(latest_path):
                    gr.Markdown("⚠️ No result file found at the expected location. Run `main.py` first.")
                else:
                    with open(latest_path, "r") as f:
                        payload_data = json.load(f)

                    # Pull the metric and baseline.
                    my_acc = float(payload_data["metrics"].get(metric_name, float('nan')))
                    baseline_acc = float(payload_data["metrics"].get(baseline_name + "_baseline", 0.0))

                    # Draw a bar chart.
                    acc_data = {
                        "metrics": [
                            {"name": "Uniform baseline", "value": baseline_acc},
                            {"name": "Hand-set optimal paths", "value": my_acc},
                        ]
                    }
                    gr.Plot(lambda accs: [(0, 0), (1, 0), (1, accs[1]), (1, accs[0]), (0, accs[0]), (0, 0)],
                            [baseline_acc, my_acc],
                            xlabel="Metric", ylabel="Accuracy", height=300)

if __name__ == "__main__":
    demo.launch()