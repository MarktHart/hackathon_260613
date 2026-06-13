import gradio as gr
import numpy as np
import torch
from agentic.experiments import benchmark_panel, load_task, results_dir
from pathlib import Path

DEVICE = "cuda"
N_HEADS = 4
VOCAB = {"pad": 0, "open": 1, "close": 2}


def compute_attention(input_ids: np.ndarray) -> np.ndarray:
    """Replicate the hand-coded stack attention from main.py for the demo."""
    tokens = torch.as_tensor(input_ids, dtype=torch.long, device=DEVICE)
    batch, seq_len = tokens.shape

    attention = torch.zeros((batch, N_HEADS, seq_len, seq_len), dtype=torch.float32, device=DEVICE)

    for b_idx in range(batch):
        seq = tokens[b_idx]
        stack = []
        pairs = []

        for pos in range(seq_len):
            tok = seq[pos].item()
            if tok == 1:
                stack.append(pos)
            elif tok == 2:
                if stack:
                    open_pos = stack.pop()
                    pairs.append((open_pos, pos))

        for i in range(seq_len):
            tok = seq[i].item()
            causal_mask = torch.arange(i + 1, device=DEVICE)

            if tok == 2:
                match_open = None
                for op, cp in pairs:
                    if cp == i:
                        match_open = op
                        break

                if match_open is not None:
                    attention[b_idx, :, i, match_open] = 0.7
                    remaining = 0.3 / len(causal_mask)
                    attention[b_idx, :, i, causal_mask] = remaining
                    attention[b_idx, :, i, match_open] = 0.7
                else:
                    uniform_val = 1.0 / len(causal_mask)
                    attention[b_idx, :, i, causal_mask] = uniform_val
            else:
                uniform_val = 1.0 / len(causal_mask)
                attention[b_idx, :, i, causal_mask] = uniform_val

    row_sums = attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    attention = attention / row_sums
    return attention.detach().cpu().numpy().astype(np.float32)


def format_sequence(tokens: np.ndarray) -> str:
    """Convert token array to readable string."""
    sym = {0: "·", 1: "(", 2: ")"}
    return "".join(sym.get(int(t), "?") for t in tokens)


def find_matching_pairs(tokens: np.ndarray) -> list[tuple[int, int]]:
    """Find (open_pos, close_pos) pairs in a single sequence."""
    stack = []
    pairs = []
    for i, t in enumerate(tokens):
        if t == 1:
            stack.append(i)
        elif t == 2:
            if stack:
                pairs.append((stack.pop(), i))
    return pairs


def plot_attention_heatmap(attn: np.ndarray, tokens: np.ndarray, head: int, layer_name: str):
    """Create a matplotlib heatmap for attention weights."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seq_len = len(tokens)
    # Only show non-PAD region
    non_pad = np.where(tokens != 0)[0]
    if len(non_pad) == 0:
        return None
    end = non_pad[-1] + 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(attn[0, head, :end, :end], cmap="Blues", vmin=0, vmax=1.0)

    # Token labels
    sym = {0: "PAD", 1: "(", 2: ")"}
    labels = [sym.get(int(t), "?") for t in tokens[:end]]
    ax.set_xticks(range(end))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(end))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title(f"{layer_name} - Head {head} (seq len {end})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def run_demo(seq_idx: int, head_idx: int):
    """Run the model on the canonical batch and return visualization for one sequence."""
    task = load_task(__file__)
    batch = task.generate(seed=42)

    if seq_idx >= len(batch.tokens):
        seq_idx = 0

    tokens = batch.tokens[seq_idx : seq_idx + 1]  # keep batch dim
    attention = compute_attention(tokens)

    # Plot heatmap
    fig = plot_attention_heatmap(attention, tokens[0], head_idx, "Stack Attention")

    # Also compute metrics for this sequence
    pairs = find_matching_pairs(tokens[0])
    seq_str = format_sequence(tokens[0])

    # Text summary
    summary_lines = [
        f"Sequence {seq_idx}: {seq_str}",
        f"Length: {len(tokens[0])}, Pairs found: {len(pairs)}",
        "",
        "Matching pairs (open, close):",
    ]
    for op, cp in pairs:
        depth = 1
        # compute depth of this pair
        open_count = sum(1 for o, c in pairs if o < op and c > cp)
        depth = open_count + 1
        attn_val = float(np.mean(attention[0, :, cp, op]))
        summary_lines.append(f"  ({op}, {cp}) depth={depth} → mean attn to match = {attn_val:.4f}")

    summary = "\n".join(summary_lines)
    return fig, summary


def list_runs():
    """List available run directories for the dropdown."""
    run_dir = results_dir(__file__)
    base = Path(run_dir).parent
    if not base.exists():
        return []
    runs = sorted([d.name for d in base.iterdir() if d.is_dir()], reverse=True)
    return runs


def load_run_metrics(run_name: str):
    """Load benchmark.json from a run directory."""
    import json
    run_dir = results_dir(__file__)
    base = Path(run_dir).parent
    bench_path = base / run_name / "benchmark.json"
    if not bench_path.exists():
        return "No benchmark.json found"
    with open(bench_path) as f:
        data = json.load(f)
    return json.dumps(data.get("metrics", data), indent=2)


# ---- Gradio app ----
with gr.Blocks() as demo:
    gr.Markdown("## Stack-like Attention in Dyck-1 Generation (pass_2)")

    with gr.Tabs():
        # ---- Demo Tab ----
        with gr.TabItem("Demo"):
            gr.Markdown(
                "Visualize the hand-coded stack attention pattern. "
                "Each closing parenthesis attends strongly (0.7 mass) to its matching opening parenthesis, "
                "with the remaining mass distributed uniformly over the causal prefix."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    seq_slider = gr.Slider(
                        minimum=0, maximum=255, value=0, step=1, label="Sequence index (0-255)"
                    )
                    head_slider = gr.Slider(
                        minimum=0, maximum=3, value=0, step=1, label="Attention head (0-3)"
                    )
                    run_btn = gr.Button("Visualize", variant="primary")
                with gr.Column(scale=2):
                    plot_out = gr.Plot(label="Attention Heatmap")
                    summary_out = gr.Textbox(label="Pairwise Attention Details", lines=20)

            run_btn.click(run_demo, inputs=[seq_slider, head_slider], outputs=[plot_out, summary_out])
            # Initial load
            demo.load(run_demo, inputs=[seq_slider, head_slider], outputs=[plot_out, summary_out])

        # ---- Benchmark Tab ----
        with gr.TabItem("Benchmark"):
            gr.Markdown("### Leaderboard and history across attempts")
            benchmark_panel(goal_dir="experiments/attention_cfg_generate")

if __name__ == "__main__":
    demo.launch()