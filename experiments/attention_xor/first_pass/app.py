import gradio as gr
import numpy as np
from agentic.experiments import (
    load_task,
    benchmark_panel,
    load_experiment,
    latest_run_dir,
)

task = load_task(__file__)

# ---- Gradio UI components ----
with gr.Blocks() as demo:
    gr.Markdown(f"# `XOR = (2 * (A ^ B) - 1)` — hand-built NumPy vs tiny MLP")

    MODE = ["hand-built", "tiny MLP"]
    with gr.Tabs():
        with gr.TabItem("Demo"):
            with gr.Row():
                # User inputs
                with gr.Block():
                    p_slider = gr.Slider(
                        0.1, 0.9, value=0.5, step=0.1, label="P(XOR=1) = P(A=1) = P(B=1)"
                    )
                    batch_button = gr.Button("Generate batch")
                # Viz area
                with gr.Block():
                    mode_dd = gr.Dropdown(MODE, label="Mode")
                    token_table = gr.DataFrame(
                        [], headers=["CLS", "A", "B", "SEP", "A", "B", "AB", "A^B"]
                    )
                    logit_plot = gr.Plot()
                    acc_display = gr.Number(label="XOR accuracy on this batch")

            def _make_tokens(p: float, n: int = 100):
                rng = np.random.default_rng(42)
                A = rng.binomial(1, p, size=(n,))
                B = rng.binomial(1, p, size=(n,))
                labels = (A ^ B).astype(np.int64)
                tokens = np.zeros((n, 4), dtype=np.int64)
                tokens[:, 0] = 0  # CLS
                tokens[:, 1] = 1 + A  # A_tok
                tokens[:, 2] = 3 + B  # B_tok
                tokens[:, 3] = 5  # SEP
                return tokens, labels, A, B

            def _compute_logits(tokens: np.ndarray, mode: str) -> tuple[np.ndarray, dict]:
                if mode == "hand-built":
                    A = (tokens[:, 1] == 2).astype(np.float32)
                    B = (tokens[:, 2] == 4).astype(np.float32)
                    feature = np.stack([A, B, A * B], axis=-1)
                    W = np.array([2.0, 2.0, -4.0])
                    b = -1.0
                    logits = (feature @ W[..., None])[:, 0] + b
                else:
                    import torch
                    from torch import nn

                    class TinyMLP(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.embed = nn.Embedding(6, 64, padding_idx=0)
                            self.proj = nn.Linear(64, 128)
                            self.relu = nn.ReLU()
                            self.head = nn.Linear(128, 1)

                        def forward(self, x):
                            x = self.embed(x)
                            x = self.proj(x.mean(dim=1))
                            x = self.relu(x)
                            return self.head(x).squeeze(-1)

                    m = TinyMLP()
                    with torch.no_grad():
                        m.embed.weight[1, 0] = 1  # 1 => [1,0,0,0,...]
                        m.embed.weight[2, 1] = 1  # 2 => [0,1,0,0,...]
                        m.embed.weight[3, 2] = 1  # 3 => [0,0,1,0,...]
                        m.embed.weight[4, 3] = 1  # 4 => [0,0,0,1,...]
                        m.proj.weight.zero_()
                        for i in range(128):
                            m.proj.weight[i, i] = 1
                        head_w = torch.zeros((1, 128))
                        head_w[0, 0] = 2  # A
                        head_w[0, 1] = 2  # B
                        head_w[0, 2] = -4  # AB (from token 4 embedding)
                        head_w[0, 3] = -1  # bias
                        m.head.weight.copy_(head_w)
                        m.head.bias.zero_()

                    logits = (
                        m(
                            torch.as_tensor(tokens, dtype=torch.int64).to(device="cpu")
                        )
                        .detach()
                        .numpy()
                    )
                return logits, {"A": A, "B": B, "AB": A * B, "XOR": (A != B).astype(np.int64)}

            def update(p: float):
                n = 100
                tokens, labels, A, B = _make_tokens(p, n)
                logits, per_token = _compute_logits(tokens, mode_dd.value)

                # Build token table rows
                rows = []
                for i in range(n):
                    row = list(tokens[i]) + [A[i], B[i], per_token["AB"][i], per_token["XOR"][i]]
                    rows.append(row)

                # Logit distribution
                fig, ax = plt.subplots()
                ax.hist(logits, bins=20, alpha=0.7, density=True)
                ax.axvline(0, color="r", linestyle="--", linewidth=1)
                ax.set_xlabel("Logit")
                ax.set_ylabel("Density")
                ax.set_title("Logit distribution (red: prediction threshold)")

                # Accuracy
                acc = float(np.mean((logits > 0) == labels))
                return (
                    rows,
                    fig,
                    acc,
                )

            batch_button.click(update, inputs=p_slider, outputs=[token_table, logit_plot, acc_display])
            mode_dd.change(  # Re-evaluate when mode changes (recompute with same batch)
                lambda mode, p: update(p), inputs=[mode_dd, p_slider], outputs=[token_table, logit_plot, acc_display]
            )

        with gr.TabItem("Benchmark"):
            # Reuse the standard benchmark panel for all runs under the goal
            panel = benchmark_panel(task.goal_dir)
            panel.render()

if __name__ == "__main__":
    demo.launch()