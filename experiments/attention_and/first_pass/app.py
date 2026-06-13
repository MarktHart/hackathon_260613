import gradio as gr
from ageneric.experiments import load_task, benchmark_panel

with gr.Blocks(title="attention_and - first_pass") as demo:
    gr.Markdown("## Attention AND Mechanism (product probe)")

    with gr.Tabs(elem_id="tab-panel"):
        with gr.TabItem("Demo"):
            gr.Markdown("This demo visualizes the product probe model's conjunction strength across feature alignment (cosine similarity).")

            cosineSlider = gr.Slider(minimum=0.0, maximum=0.9, step=0.2, label="Cos(e_A, e_B)")
            with gr.Row():
                bothLogit = gr.Number(label="Logit (both)")
                aLogit = gr.Number(label="Logit (A only)")
                bLogit = gr.Number(label="Logit (B only)")
                noneLogit = gr.Number(label="Logit (none)")
                
            def update_logits(cosine):
                batch = load_task(__file__).generate()
                ci = np.where(np.isclose(batch.cosines, cosine))[0][0]
                rec = {
                    "logit_both": batch.logit_both[ci],
                    "logit_a": batch.logit_a[ci],
                    "logit_b": batch.logit_b[ci],
                    "logit_none": batch.logit_none[ci],
                    "n_trials": 200
                }
                return (
                    float(rec["logit_both"]),
                    float(rec["logit_a"]),
                    float(rec["logit_b"]),
                    float(rec["logit_none"])
                )
            
            cosineSlider.change(fn=update_logits, inputs=cosineSlider, outputs=[bothLogit, aLogit, bLogit, noneLogit])

        # Benchmark tab
        gr.TabItem("Benchmark").render(benchmark_panel)

# Launch the app
if __name__ == "__main__":
    demo.launch()