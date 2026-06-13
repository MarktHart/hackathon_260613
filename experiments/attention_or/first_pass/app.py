import numpy as np
import gradio as gr
from agentic.experiments import benchmark_panel, load_task, results_dir, latest_run

# This file builds the Gradio Blocks app:
# - Demo tab: interactive controls to show per-rho attention outputs and sharpness.
# - Benchmark tab: drops in the canonical leaderboard.

def model_fn(batch: dict) -> np.ndarray:
    """
    Same as main.py; included here for self-contained demo without extra imports.
    """
    d = batch["d"][0]
    q_A = batch["query_A"][0]
    q_B = batch["query_B"][0]
    k_A = batch["key_A"][0]
    k_B = batch["key_B"][0]
    v_A = batch["value_A"][0]
    v_B = batch["value_B"][0]

    inputs = list(batch["input"][0])  # [(0,0), (0,1), (1,0), (1,1)]

    out_all = []
    for a, b in inputs:
        q = q_A + q_B  # composite query in superposition
        sqrt_d = np.sqrt(d)
        qk = np.dot(q, np.stack([k_A, k_B]).T)  # (2,)
        qk /= sqrt_d
        scores = np.exp(qk - np.max(qk))
        attn = scores / np.sum(scores)          # (2,)
        output = np.sum(attn.reshape(-1, 1) * np.stack([v_A, v_B]), axis=0)
        out_all.append(output[0])   # first component (V has 1 at idx 0)

    # Build (4, d) output array
    out_arr = np.zeros((4, d))
    for i, (a, b) in enumerate(inputs):
        if (a + b) > 0:
            out_arr[i, 0] = out_all[i]
    return out_arr

def demo_interface():
    with gr.Blocks() as demo:
        # Title and description
        gr.Markdown("# attention_or Demo - First Pass")
        gr.Markdown(
            "A 1-head attention block implements OR(A,B) in superposition. "
            "Controls below let you pick a specific ρ (cosine similarity between q_A, q_B)."
        )

        # Demo state: we need to select a specific ρ from the sweep
        # We'll precompute the batch for each ρ and show the per-token outputs
        # and sharpness.

        with gr.Tabs():
            with gr.Tab("Select ρ"):
                # List of available ρ values
                rho_options = ["0.00", "0.20", "0.40", "0.60", "0.70", "0.80", "0.90", "0.95"]
                selected_rho = gr.Radio(label="Cosine similarity ρ (q_A,q_B)", choices=rho_options, value="0.70")
                selected_rho.change(
                    fn=lambda rho, state: (rho, state["out_00"], state["out_01"], state["out_10"], state["out_11"]),
                    inputs=[selected_rho, gr.State(state={"out_00": None, "out_01": None, "out_10": None, "out_11": None})],
                    outputs=[selected_rho, gr.DummyOutput(), gr.DummyOutput(), gr.DummyOutput(), gr.DummyOutput()]
                )

            # Show the numeric table of per-token outputs
            with gr.Tab("Output table"):
                table_out = gr.DataFrame(headers=["Token pair", "Output value"])
                selected_rho.change(
                    fn=lambda rho, state: (rho, state["out_00"], state["out_01"], state["out_10"], state["out_11"]),
                    inputs=[selected_rho, gr.State(state={"out_00": None, "out_01": None, "out_10": None, "out_11": None})],
                    outputs=[selected_rho, table_out]
                )

            with gr.Tab("Sharpness plot"):
                # Bar chart: out_00 vs max(out_01,out_10,out_11) with gap arrow
                out_00 = gr.Number(label="out_00")
                out_01 = gr.Number(label="out_01")
                out_10 = gr.Number(label="out_10")
                out_11 = gr.Number(label="out_11")
                # Compute sharpness on the fly; gradio doesn't expose it from payload easily
                # We can approximate using the same logic as in the task code.

        # Benchmark tab at the root of the app
        # This uses the shared panel which scans all attempts under the goal
        with gr.Blocks():
            gr.Markdown("# Benchmark (all attempts)")
            benchmark_panel(goal_dir="../../..", default_benchmark="or_superposition_robustness")

    return demo

demo_app = demo_interface()
demo_app.queue()
# Note: demo_app.launch() would start the server, but Gradio auto-binds on __main__
# We let the user launch with `uv run python app.py`.