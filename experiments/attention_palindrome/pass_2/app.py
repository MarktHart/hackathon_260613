import gradio as gr
import numpy as np

from agentic.experiments import load_task, benchmark_panel, results_dir

task = load_task(__file__)
run_dir = results_dir(__file__)

demo = gr.Blocks()
with demo:
    with gr.Tabs():
        # Demo tab
        with gr.Tab("Demo — Mirror Comparison Visualisation"):
            gr.Markdown(
                "A narrow transformer trained to detect perfect palindromes from near-palindromes. "
                "It learns a single attention head that compares mirrored positions `i` and `L-1-i`, then folds the per-position agreement signal into a final palindrome score.\n"
                "Choose a seed to generate a palindrome and a set of corrupted negatives; the demo shows the raw palindrome score per sequence and the attention heatmaps for the mirror pair."
            )
            with gr.Row():
                seed_input = gr.Number(label="Seed (int)", value=42, info="Deterministic batch for demo")
                btn_compute = gr.Button("Generate & Compute Scores")
            with gr.Row():
                seed_output = gr.Code(label="Palindrome score (one scalar per sequence)", language="textual")
            with gr.Row():
                attn_img = gr.Image(label="Attention heatmap for the selected sequence", type="numpy", width=400, height=200)
            with gr.Row():
                seq_display = gr.Label(label="Selected sequence (index)", format="json")

            def compute_demo(seed: int):
                batch_np = task.generate(seed)
                B, L = batch_np.tokens.shape
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # build a minimal model + train
                model = attention_head_base = torch.nn.ModuleList([
                    torch.nn.Embedding(num_embeddings=256, embedding_dim=256),
                    torch.nn.Embedding(num_embeddings=16, embedding_dim=256)
                ]).to(device).eval()
                model[0].load_state_dict(torch.load("experiments/attention_palindrome/pass_2/checkpoint.pt"))
                # train_model(model)

                # tokens -> PyTorch
                tokens = torch.as_tensor(batch_np.tokens).to(device)
                tokens = tokens[:, :16]            # enforce length
                pos_idx = torch.arange(L).to(device)

                # compute hidden states and final score
                with torch.no_grad():
                    emb_tok = F.embedding(tokens, model[0].weight)               # (B, L, D)
                    emb_pos = F.embedding(pos_idx.expand(B, L), model[1].weight)   # (B, L, D)
                    x = emb_tok + emb_pos

                    # we need the attention weights for each token position
                    qkv = model.blocks[0][0](x)     # (B, L, 3)   # merged attn: shape (B, L, C)
                    attn = F.softmax(model.blocks[0][1](qkv), dim=-1)   # (B, L, C)
                    # attention per query position (over key positions)
                    attn_weight = attn[:, :, 2]   # (B, L)

                    # palindrome score per sequence (single head)
                    palindrome_logits = torch.einsum('...nd,dc->...n', x, model.palindrome_head.weight)
                    out = model_mlp_head(palindrome_logits)   # (B, 1)
                    scores = out.squeeze_(-1).cpu().numpy()

                # pick a random sequence to visualise (0-indexed)
                vis_i = np.random.randint(B)

                # construct the heatmaps
                # 2 rows per sequence: attn[i] across keys, attn[j] across keys (where j = L-1-i)
                img = np.zeros((50, 200))
                for k in range(L):
                    # column 0: position label 0-L
                    img[:, 20*L + k] = np.minimum(1.0, attn_weight[vis_i, k] * 10) if k < 16 else 0
                # add a vertical center divider
               	img[:, L*10] += 0.9 if vis_i < B//2 else 0.1
                return f"Score array: {scores[:5]}...", np.clip(img, 0, 1), batch_np.tokens[vis_i].tolist()

            btn_compute.click(
                fn=compute_demo,
                inputs=[seed_input],
                outputs=[seed_output, attn_img, seq_display],
                api_name="compute_demo"
            )

        # Benchmark tab
        with gr.Tab("Benchmark") as bench_tab:
            benchmark_panel(task.goal_dir)

if __name__ == "__main__":
    demo.launch()