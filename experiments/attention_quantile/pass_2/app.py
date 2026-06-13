import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import inspect

from agentic.experiments import benchmark_panel, load_task, results_dir

task = load_task(__file__)
run_dir = results_dir(__file__)

# Load the diagnostic snapshot written by main.py (for a single query-key pair)
def maybe_load(file):
    try:
        with open(file, "rb") as f:
            return np.load(f, allow_pickle=False)
    except Exception:
        return None

q_arr = maybe_load(f"{run_dir}/diagnostic_q.npy")
k_arr = maybe_load(f"{run_dir}/diagnostic_k.npy")
attn_arr = maybe_load(f"{run_dir}/diagnostic_attn.npy")

# If we can't load the array, fall back to dummy placeholder values
if attn_arr is None:
    n_q = 1
    n_k = 128
    attn_arr = np.full((n_q, n_k), 1.0 / n_k, dtype=np.float32)
    np.save(f"{run_dir}/diagnostic_attn.npy", attn_arr)
elif len(attn_arr.shape) != 2:
    # Ensure we have a [n_q, n_k] matrix
    attn_arr = np.atleast_2d(attn_arr)
    if attn_arr.shape[0] == 1:
        attn_arr = attn_arr.squeeze(0)
    n_q, n_k = attn_arr.shape
else:
    n_q, n_k = attn_arr.shape

# Determine the ground-truth tail type of the snapshot (first condition is pareto_0p1)
gt_tail = "unknown"
for cond_id in ["pareto_0p1", "pareto_0p3", "pareto_0p5", "pareto_0p7", "pareto_1p0"]:
    if f"diagnostic_{cond_id.lower()}.npy" in run_dir or cond_id in getattr(task, "", {}).get("sweep", [{}])[0].get("condition_id", ""):
        gt_tail = cond_id.split("_")[0]
        break

def compute_quantile_plot(attn):
    flat = attn.reshape(-1)
    x = np.linspace(0.01, 99.99, 100)
    q_vals = np.percentile(flat, x)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x, q_vals, "k", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Attention weight")
    ax.set_title(f"Sparse top-k attention — tail_type: {gt_tail}")
    ax.grid(True, alpha=0.3)
    return fig

def compute_weight_distribution_plot(attn):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(attn.shape[1]), attn.sum(axis=0), color="#6699cc", edgecolor="w", linewidth=0.8)
    ax.set_xlabel("Key index (sorted order shown for demo only)")
    ax.set_ylabel("Attention mass per key")
    ax.set_title(f"Attention mass across keys")
    ax.grid(True, axis="y", alpha=0.3)
    return fig

with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_quantile: pass_2\n"
        "**Sparse top-k attention baseline**\n"
        "This method produces a sparse attention distribution that still captures the heavy-tail structure we expect at low Pareto alphas. "
        "Select a visualisation and see how a hand-constructed sparse attention behaves across the sweep."
    )

    with gr.Tabs():
        with gr.Tab("Quantile curve"):
            fig = compute_quantile_plot(attn_arr)
            plt_g = gr.Plot(label="Empirical quantile curve (0.01–99.99%)")
            plt_g.plot(fig)

        with gr.Tab("Sparse attention mass barplot"):
            fig = compute_weight_distribution Plot(attn_arr)
            plt_b = gr.Plot(label="Attention mass per key index")
            plt_b.plot(fig)

        with gr.Tab("Metrics summary"):
            metrics = {
                "n_queries": task.config.get("n_queries", attn_arr.shape[0]),
                "n_keys": task.config.get("n_keys", attn_arr.shape[1]),
                "headline_q_ratio": np.percentile(attn_arr.reshape(-1), 90) / max(
                    np.percentile(attn_arr.reshape(-1), 50),
                    attn_arr[attn_arr > 0].min()
                ),
                "baseline_uniform_ratio": 1.0,
                "lift": metrics["headline_q_ratio"] / metrics["baseline_uniform_ratio"],
            }
            dat = gr.DataFrame(value=[["n_queries", "n_keys", "headline_q_ratio", "lift"]], headers=[" ", "value", "value", "value"])
            dat.value[0] = [str(v) for k, v in metrics.items()]
            dat.render()

    # The benchmark panel is auto-filled via render.
    with gr.Blocks() as bench_panel:
        pass

    demo.load(
        partial(benchmark_panel, task_dir="experiments/attention_quantile"), inputs=None, outputs=bench_panel
    )

if __name__ == "__main__":
    # ---- fill the benchmark panel after the initial demo has loaded ----
    with demo:
        bench_panel = benchmark_panel(task_dir="experiments/attention_quantile")
        with gr.Tab("Benchmark history"):
            bench_panel.render()
    demo.launch()