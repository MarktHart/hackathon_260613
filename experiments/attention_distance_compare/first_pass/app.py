import gradio as gr
from agentic.experiments import load_task, benchmark_panel
import numpy as np

with gr.Blocks() as demo:
    gr.Markdown("# Attention Distance Compare - First Pass")
    
    # Demo tab
    gr.Markdown("### Distance-Sensitive Attention Model Demo")
    gr.Markdown("This app demonstrates an attention model that depends on positional distance between query and key tokens.")
    
    # Simple demo that shows the attention pattern
    with gr.Tabs():
        # Distance bins
        distance_bins = [0, 1, 2, 4, 8, 16, 32]
        
        with gr.Blocks():
            gr.Markdown("#### Attention Head Pattern")
            with gr.Row():
                with gr.Column():
                    seq_len = gr.Slider(minimum=4, maximum=64, value=20, step=1, label="Sequence Length")
                with gr.Column():
                    lambda_param = gr.Slider(minimum=1, maximum=20, value=6, step=1, label="Distance Decay Parameter")
            
            head_pattern = gr.Image(label="Attention Head Weight Distribution")
            gr.Markdown("The image shows an attention head where the strength of attention falls off with distance from the query token.")
            
            def update_head_pattern(seq_len, lambda_param):
                # Generate a visualization of the attention pattern
                attn = np.zeros(( seq_len, seq_len))
                causal_mask = np.tril(np.ones(( seq_len, seq_len)))
                
                for i in range(seq_len):
                    for j in range(i + 1):
                        dist = i - j
                        if dist == 0:
                            attn[i, j] = 1.0
                        else:
                            attn[i, j] = np.exp(-dist / lambda_param)
                
                # Convert to image (query on y-axis, key on x-axis)
                img = np.flip(attn, axis=0)  # Flip rows for visualization
                return np.array(img)
            
            seq_len.input(update_head_pattern, inputs=[seq_len, lambda_param], outputs=[head_pattern])
            lambda_param.input(update_head_pattern, inputs=[seq_len, lambda_param], outputs=[head_pattern])
            
            # Initial display
            head_pattern.value = update_head_pattern(seq_len.value, lambda_param.value)
        
        # Benchmark tab
        with gr.Blocks():
            gr.Label(value="Benchmark History and Comparison")
            benchmark_panel(load_task(__file__))

if __name__ == "__main__":
    demo.launch()