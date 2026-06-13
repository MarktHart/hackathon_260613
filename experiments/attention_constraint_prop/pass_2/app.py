import gradio as gr
import torch
import numpy as np
from pathlib import Path

from agentic.experiments import benchmark_panel, results_dir

# Import the task and model components
import experiments.attention_constraint_prop.task as task_module
from experiments.attention_constraint_prop.pass_2.main import (
    BracketTransformer, DEVICE, SEQ_LEN, VOCAB_SIZE, N_FILLER,
    OPEN_A, CLOSE_A, OPEN_B, CLOSE_B, N_LAYERS, N_HEADS, D_MODEL, D_HEAD, DROPOUT,
    generate_batch, constraint_loss, make_model_fn
)

# ---- Load trained model for demo --------------------------------------------
def load_trained_model():
    """Load the most recently trained model checkpoint."""
    run_dir = results_dir(__file__)
    checkpoint_path = Path(run_dir) / "model.pt"
    if not checkpoint_path.exists():
        return None
    
    model = BracketTransformer().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


# ---- Demo computation -------------------------------------------------------
def run_demo(seed: int):
    """Run demo on a single batch with given seed, return DataFrame and attention viz."""
    model = load_trained_model()
    if model is None:
        # Return empty dataframe with message
        import pandas as pd
        df = pd.DataFrame({
            "distance": [],
            "n_entries": [],
            "mean_alignment": [],
            "max_alignment": [],
            "best_head_layer": [],
            "best_head_head": [],
        })
        return df, "No trained model found. Run main.py first to train."
    
    model_fn = make_model_fn(model)
    
    # Generate batch with given seed
    batch = task_module.generate(seed=seed)
    
    # Evaluate
    payload = task_module.evaluate(model_fn, batch=batch)
    
    # Build DataFrame for display
    import pandas as pd
    rows = []
    for rec in payload["sweep"]:
        rows.append({
            "distance": rec["distance"],
            "n_entries": rec["n_entries"],
            "mean_alignment": round(rec["mean_alignment"], 6),
            "max_alignment": round(rec["max_alignment"], 6),
            "best_head_layer": rec["best_head"]["layer"],
            "best_head_head": rec["best_head"]["head"],
        })
    df = pd.DataFrame(rows)
    
    # Compute fidelity
    baseline = 1.0 / SEQ_LEN
    canonical_rec = next((r for r in payload["sweep"] if r["distance"] == task_module.CANONICAL_DISTANCE), None)
    if canonical_rec:
        fidelity = canonical_rec["max_alignment"] / baseline
        summary = (f"Constraint Propagation Fidelity (dist={task_module.CANONICAL_DISTANCE}): "
                   f"{fidelity:.2f}x baseline (baseline={baseline:.4f}, "
                   f"best_head alignment={canonical_rec['max_alignment']:.4f})")
    else:
        summary = "Canonical distance not found in sweep."
    
    return df, summary


# ---- Attention heatmap for a specific head ----------------------------------
def get_attention_heatmap(layer: int, head: int, seed: int):
    """Return attention matrix for a specific layer/head on a demo sequence."""
    model = load_trained_model()
    if model is None:
        return None, "No trained model found."
    
    # Generate one sequence
    batch = task_module.generate(seed=seed, num_sequences=1)
    input_ids = batch.input_ids[0:1]  # [1, S]
    
    model_fn = make_model_fn(model)
    attn = model_fn(input_ids)  # [1, L, H, S, S]
    
    # Get attention for specified layer/head
    attn_matrix = attn[0, layer, head]  # [S, S]
    
    # Also return the tokens for labeling
    tokens = input_ids[0].tolist()
    token_labels = []
    for t in tokens:
        if t == OPEN_A: token_labels.append("[A")
        elif t == CLOSE_A: token_labels.append("A]")
        elif t == OPEN_B: token_labels.append("[B")
        elif t == CLOSE_B: token_labels.append("B]")
        else: token_labels.append(str(t))
    
    return attn_matrix, token_labels


# ---- Gradio App -------------------------------------------------------------
with gr.Blocks(title="Attention Constraint Propagation - pass_2") as demo:
    gr.Markdown("""
    # Attention Constraint Propagation: Trained Transformer (pass_2)
    
    **Question:** Do attention heads learn to propagate bracket constraints (matched open/close pairs) 
    across positions, and how does fidelity vary with positional distance?
    
    This attempt trains a small 2-layer, 8-head transformer with a loss that directly 
    encourages attention heads to attend from each bracket to its partner.
    """)
    
    with gr.Tab("Demo"):
        with gr.Row():
            seed_input = gr.Number(value=0, label="Batch Seed", precision=0)
            run_btn = gr.Button("Run Demo", variant="primary")
        
        with gr.Row():
            summary_out = gr.Textbox(label="Summary", interactive=False)
        
        df_out = gr.DataFrame(
            label="Alignment by Distance",
            headers=["distance", "n_entries", "mean_alignment", "max_alignment", "best_head_layer", "best_head_head"],
            datatype=["number", "number", "number", "number", "number", "number"],
            interactive=False
        )
        
        gr.Markdown("### Attention Heatmap (Query → Key)")
        with gr.Row():
            layer_sel = gr.Dropdown(choices=[0, 1], value=0, label="Layer")
            head_sel = gr.Dropdown(choices=list(range(N_HEADS)), value=0, label="Head")
            heatmap_seed = gr.Number(value=0, label="Sequence Seed", precision=0)
            heatmap_btn = gr.Button("Show Heatmap")
        
        heatmap_plot = gr.Plot(label="Attention Weights")
        
        # Event handlers INSIDE the Blocks context
        run_btn.click(
            fn=run_demo,
            inputs=[seed_input],
            outputs=[df_out, summary_out]
        )
        
        def plot_heatmap(layer, head, seed):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            attn_matrix, token_labels = get_attention_heatmap(layer, head, seed)
            if attn_matrix is None:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "No model found", ha='center', va='center')
                return fig
            
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(attn_matrix, cmap='Blues', vmin=0, vmax=attn_matrix.max())
            ax.set_title(f"Layer {layer}, Head {head} — Attention Weights")
            ax.set_xlabel("Key Position")
            ax.set_ylabel("Query Position")
            ax.set_xticks(range(len(token_labels)))
            ax.set_xticklabels(token_labels, rotation=90, fontsize=6)
            ax.set_yticks(range(len(token_labels)))
            ax.set_yticklabels(token_labels, fontsize=6)
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            return fig
        
        heatmap_btn.click(
            fn=plot_heatmap,
            inputs=[layer_sel, head_sel, heatmap_seed],
            outputs=[heatmap_plot]
        )
        
        # Auto-run on load
        demo.load(
            fn=run_demo,
            inputs=[seed_input],
            outputs=[df_out, summary_out]
        )
    
    with gr.Tab("Benchmark"):
        goal_dir = "experiments/attention_constraint_prop"
        benchmark_panel(goal_dir)

if __name__ == "__main__":
    demo.launch()