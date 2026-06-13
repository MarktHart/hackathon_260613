import gradio as gr
import json
import pathlib

from agentic.experiments import goal_dir, benchmark_panel, results_dir

# Load goal metadata
goal_path = goal_dir(__file__)
run_dir = results_dir(__file__)
run_json = run_dir / "run.json"
if not run_json.exists():
    raise FileNotFoundError(f"No run.json at {run_json}")

run = json.loads(run_json.read_text())
attempt = pathlib.Path(run["attempt_path"]).name
benchmark_dir = goal_path / " benchmarks"   # Note: typo in goal name, adjust as needed

######################
### DEMO VIEWS       ###
######################

with gr.Blocks() as demo:
    gr.Markdown(f"# First-pass attention-linear-sum attempt `{attempt}`")
    gr.Markdown("### Visualisation")
    # Add visual components here that illustrate the hand-set circuit behaviour.
    gr.Markdown("#### Why it works (hand-built intuition)"
               "1. Query projection isolates feature dims 0 and 1."
               "2. Key projection pairs α with x1, β with x2."
               "3. Softmax attention picks the right scalar coefficient."
               "4. Output projection adds the two weighted terms."

    # Optional: a simple table showing performance across (α,β) slices
    try:
        benchmark_path = benchmark_dir / "benchmark.json"
        bench = json.loads(benchmark_path.read_text())
        metrics = bench.get(attempt, {})
        # Show a concise table of per-coefficient R²
        table_rows = []
        for k in metrics:
            if k.startswith("linear_combination_r2_alpha_"):
                parts = k.split("_")
                alpha = parts[3]
                beta_part = parts[5] if len(parts) > 5 else "0"  # placeholder
                # Format alpha/beta nicely as 1p0 etc.
                def deFmt(v):
                    if v == "0p0" or not v: v="0"
                    else: v = v.replace("p",".")
                    if "m" in v: return "-" + deFmt(v[1:])
                    if v.startswith("0p"): return "0"
                    return v
                a = deFmt(alpha)
                b = deFmt(beta_part)
                row = f"`α={a}, β={b}` | {metrics[k]:.3f}"
                table_rows.append(row)
        demo.markdown(f"#### Per-coefficient performance\n\n| α,β | R² |\n|---:|:---|\n" + "\n".join(table_rows))
    except Exception as e:
        demo.markdown(f"#### Metric summary unavailable\n\n```{e}```")

    gr.Markdown("---")
    gr.Markdown("## Benchmark tab")
    gr.Markdown("The following panel shows the same attempt across all runs and compares to previous attempts.")

# BENCHMARK TAB (uses agentic's built-in)
benchmark_tab = benchmark_panel(goal_path, "attention_linear_sum", default_attempt=attempt)

with gr.Blocks() as demo:
    # Demo UI is empty; only Benchmark tab matters
    pass   # demo already defined above

if __name__ == "__main__":
    demo.launch()