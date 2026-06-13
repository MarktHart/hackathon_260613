import gradio as gr
import numpy as np

from agentic.experiments import benchmark_panel, load_task, results_dir

task = load_task(__file__)
run_dir = results_dir(__file__)

with open(f"{run_dir}/diagnostic_attn.npy", "rb") as f:
    attn_arr = np.load(f, allow_pickle=False)
with open(f"{run_dir}/diagnostic_true_q.npy", "rb") as f:
    true_q_arr = np.load(f, allow_pickle=False)
with open(f"{run_dir}/diagnostic_pred_q.npy", "rb") as f:
    pred_q_arr = np.load(f, allow_pickle=False)

def demo_row(q_levels, head_idx, attn, true_q, pred_q):
    # Render a row of the demo: attention weights (as a bar plot), true quantile,
    # predicted quantile, and error. head_idx is for display only.
    fig, ax = gr.Plot(
        x=[i for i in range(task SEQ_LEN)], y=attn[head_idx], title=f"Head {head_idx + 1} (α ~ {np.quantile(attn[head_idx], 0.5):.3f})"
    ).get_plot()
    ax.set_xlabel("Position in key sequence")
    ax.set_ylabel("Weight")
    ax.set_ylim(bottom=0, top=1.02)

    metrics = {
        f"Quantile {level*100:.0f}%": pred_q[head_idx, i] - true_q[head_idx, i]
        for i, level in enumerate(q_levels)
    }

    return fig, metrics

with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_quantile: first_pass\n"
        "Baseline that just reads the empirical quantiles of each attention row.\n"
        "Select any head and see its weight distribution (blue bar) and the MAE per quantile."
    )
    with gr.Row():
        head_dd = gr.Dropdown(
            choices=[i for i in range(task.NUM_HEADS)], value=0, label="Show head"
        )
        q_levels = ["0.1", "0.25", "0.5", "0.75", "0.9", "0.95", "0.99"]
        q_mse = gr.DataFrame(
            headers=["Quantile"] + q_levels,
            datatype=["str"] + ["number"] * len(q_levels),
            value=[[q] + [0.0] * len(q_levels) for q in [""] + q_levels],
        )
    attn_plot = gr.Plot()
    head_dd.change(
        demo_row,
        inputs=[q_levels, head_dd, attn_arr[0], true_q_arr[0], pred_q_arr[0]},
        outputs=[attn_plot, q_mse],
    )
    demo.load(demo_row, inputs=[q_levels, head_dd, attn_arr[0], true_q_arr[0], pred_q_arr[0]], outputs=[attn_plot, q_mse])

    with gr.Blocks().style(height="300") as bench:
        pass  # demo.load will fill this once benchmarks are present.

if __name__ == "__main__":
    # ---- fill the benchmark panel after the initial demo has loaded ----
    with demo:
        bench_panel = benchmark_panel(task_dir="experiments/attention_quantile")
        with gr.Tab("Benchmark"):
            bench_panel.render()
    demo.launch()