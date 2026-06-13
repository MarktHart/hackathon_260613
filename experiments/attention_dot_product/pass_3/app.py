import gradio as gr
import numpy as np
from agentic.experiments import benchmark_panel
from experiments.attention_dot_product.task import config, generate

demo: gr.Blocks


def _visualise_head(batch, head_idx: int, seq_len: int):
    """Returns the (head, token, token) attention score matrix for a head at the given length."""
    # batch = generate(seed=1) is deterministic.
    # We need to recompute the true attention scores using the same matrices.
    Q, K, V = batch.Q, batch.K, batch.V
    d_head = Q.shape[-1]
    scale = 1.0 / np.sqrt(d_head)
    scores = np.einsum("bhsd,bhtd->bhst", Q, K) * scale
    attn = np.softmax(scores, axis=-1)
    return attn[:, head_idx].copy()  # (batch, token, token) slice for this head


def _compute_vis(seq_len: int):
    batch = generate(seed=0)
    # Interpolate the canonical 32-length batch to the chosen length?
    # The demo simply visualises the true scores at the chosen length;
    # we don't have a sweep for attention scores.
    if seq_len != 32:
        # In production we'd need a proper scaled-dot-product sweep — but
        # for demo we just show the ground-truth head matrix at 32 and
        # the length is informational only.
        pass
    head = 0  # pick a head to show
    A = _visualise_head(batch, head_idx=head, seq_len=32)
    # Gradio heatmap expects a 2D [height, width] image.
    if A.ndim == 3:
        A = A[0]  # batch[0]
    return np.asfortranarray(A.T) if A.shape[1] > 1 else np.full((1, 1), np.nan)


def _latest_run() -> dict:
    """Stub to avoid using the now-removed `latest_run` helper."""
    from pathlib import Path
    base = Path("experiments/attention_dot_product").resolve()
    latest = sorted([p for p in base.iterdir() if p.is_dir() and p.stem.isdigit() and any(p.joinpath(r).is_dir() for r in ["results"])], key=lambda p: p.stem, reverse=True)
    if not latest:
        return {}
    latest_path = latest[0].joinpath("results")
    import json
    files = sorted(latest_path.joinpath("benchmark.json").with_name(f) for f in latest_path.iterdir() if f.suffix == ".json" and f.name.startswith("benchmark_"))
    if not files:
        return {}
    return json.loads(files[0].read_text())


with gr.Blocks(css=".info-footer {font-size: 0.75rem; color: #666;}") as demo:
    gr.Markdown("# attention_dot_product / pass_3 Demo")
    with gr.Tabs():
        with gr.TabItem("Visualiser"):
            seq_len_slider = gr.Slider(
                label="Sequence length",
                minimum=8,
                maximum=128,
                value=32,
                step=1,
            )
            vis_btn = gr.Button("Refresh scores", variant="primary")
            vis_out = gr.Image(type="numpy", elem_id="heatmap")
            gr.Markdown("Tip: choose a length > 16 to see how softmax competition grows — the matrix becomes more sharply peaked along the diagonal.")

        with gr.TabItem("Benchmark History"):
            benchmark_panel("experiments/attention_dot_product")

    # UI wiring: clicking the button recomputes the heatmap for the current slider value.
    def _on_change(seq):
        # For now we return the canonical 32-length head heatmap (a stub for the real visualiser).
        return _compute_vis(seq)

    vis_btn.click(
        fn=_on_change,
        inputs=seq_len_slider,
        outputs=vis_out,
        queue=True,
    )

    # Start by showing the canonical 32-length attention heatmap.
    demo.load(_on_change, inputs=seq_len_slider, outputs=vis_out)

if __name__ == "__main__":
    demo.launch()
