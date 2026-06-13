import gradio as gr
from agentic.experiments import benchmark_panel, load_task

import numpy as np

# ---- Helper to load a single run's benchmark.json (cached) ----
_BENCHMARK_CACHES = {}


def _load_benchmark_json(dirname):
    """Read benchmark.json from a results directory, caching per directory."""
    if dirname not in _BENCHMARK_CACHES:
        try:
            with open(dirname + "/benchmark.json", encoding="utf-8") as f:
                _BENCHMARK_CACHES[dirname] = json.load(f)
        except FileNotFoundError:
            _BENCHMARK_CACHES[dirname] = None
    return _BENCHMARK_CACHES[dirname]


# ---- Demo component: per-head attention heatmap ----
def _show_attention_heatmap(payload):
    """Return Gradio Table rows that summarise each head's attention pattern."""
    rows = [
        ["Layer", "Head", "Level", "Local concentration", "Chunk concentration",
         "Superchunk concentration", "Entropy (nats)"]
    ]

    for rec in payload["sweep"]:
        # Determine level based on head_idx (0–2) as defined in _attn_head_forward
        level = "local"    if rec["head"] in [0, 3, 4] else \
                "chunk"   if rec["head"] in [1, 5, 6] else \
                "superchunk" if rec["head"] == 2 else "unknown"

        rows.append([
            rec["layer"],
            rec["head"],
            level,
            f"{rec['local_concentration']:.3f}",
            f"{rec['chunk_concentration']:.3f}",
            f"{rec['superchunk_concentration']:.3f}",
            f"{rec['entropy']:.3f}"
        ])

    return rows


# ---- Core Gradio app (one block) ----
with gr.Blocks() as demo:
    gr.Markdown(f"# attention_hierarchical_pool – pass_2")
    gr.Markdown(
        "**What's being shown**: a hand-built attention-only transformer with 12 layers and 8 heads per layer."
        "\nHeads 0 (local), 1 (chunk), 2 (superchunk) implement the three hierarchical pooling levels."
        "\nHeads 3–7 are dummy flat-attention heads to fill the 8-head shape required by the goal."
    )

    # Demo tab: show the most recent run from results/
    with gr.Blocks() as demo_tab:
        gr.Markdown("## Demo – latest run")
        out_dir = gr.State(None)               # store chosen results directory
        table = gr.Table(
            show_index=False,
            col_count=7,
            wrap=False,
            label="Per-head concentrations and entropy"
        )
        with gr.Row(equal_size="md") as button_row:
            refresh_btn = gr.Button("Refresh table from latest run", variant="primary")
        with gr.Row(equal_size="sm") as note_row:
            with gr.Column(scale=3):
                gr.Markdown(
                    "The table shows, for each (layer, head),:"
                    "\n- **Local concentration**: mass inside the ±2 token window around the query."
                    "\n- **Chunk concentration**: mass inside the 16-token chunk containing the query."
                    "\n- **Superchunk concentration**: mass inside the 4-chunk (64-token) region containing the query."
                    "\n- **Entropy**: attention entropy in nats."
                )
            with gr.Column(scale=1):
                # dummy placeholder to keep the row balanced visually
                pass

        # Refresh logic
        @refresh_btn.click(
            fn=lambda: out_dir,
            inputs=[out_dir],
            outputs=[out_dir],
            queue=False  # sync to update immediately
        )
        def _RefreshTable(d, _out_dir):
            """Refresh the table after directory change."""
            _out_dir.value = d
            return d

        # When refresh_btn is released, run the table update
        refresh_btn.click(
            fn=_update_table_contents,
            inputs=out_dir,
            outputs=table
        )

    # Benchmark tab: reusable panel
    with gr.Blocks() as bench_tab:
        bench_panel = benchmark_panel("../../..")

    # Tabbed UI
    tabs = gr.Tabs()
    demo_tab.title = "Demo"
    bench_tab.title = "Benchmark"
    tabs.add瘤(demo_tab, "Demo")
    tabs.add瘤(bench_tab, "Benchmark")

    # Initial UI fill: scan most recent run
    import os, json, time
    most_recent = None
    for subdir in sorted(os.listdir("results"), reverse=True):
        if subdir.startswith("202"):   # assume UTC timestamp naming
            latest_path = os.path.join("results", subdir, "benchmark.json")
            if os.path.exists(latest_path):
                most_recent = os.path.join("results", subdir)
                break

    demo_tab.title = "Demo"
    refresh_btn.click(fn=_refresh_button, inputs=[], outputs=out_dir)
    @out_dir.change(
        fn=_load_headline_and_update_ui,
        inputs=out_dir,
        outputs=[table, headline_panel],
        queue=False
    )
    def _load_headline_and_update_ui(d):
        if d is None:
            return gr.Table.update(visible=False), ""
        try:
            bm = _load_benchmark_json(d)
            if bm is None:
                raise FileNotFoundError
            tab_rows = _show_attention_heatmap(bm)
            headline = f"Headline median hierarchical robustness: {bm.get('hierarchical_robustness_canonical', 'N/A'):.3f}"
            return gr.Table.update(value=tab_rows, visible=True), headline
        except Exception as e:
            print("Error loading benchmark:", e)
            return gr.Table.update(visible=False), "Failed to load benchmark.json"

    # Provide a way to select any run
    dropdown = gr.Dropdown(
        label="Select run",
        choices=[f"run-{name}" for name in sorted(os.listdir("results"), reverse=True)]
    )
    dropdown.change(
        fn=_select_run,
        inputs=dropdown,
        outputs=out_dir
    )


def _select_run(value):
    "Convert 'run-20260613T140146Z' back to results/20260613T140146Z"
    if not value or not value.startswith("run-"):
        return None
    name = value[len("run-"):]
    return os.path.join("results", name)


def _update_table_contents(dirname):
    if dirname is None:
        return gr.Table.update(visible=False)
    try:
        payload = _load_benchmark_json(dirname)
        if payload is None:
            raise FileNotFoundError
        rows = _show_attention_heatmap(payload)
        return gr.Table.update(value=rows, visible=True)
    except Exception:
        return gr.Table.update(visible=False)


if __name__ == "__main__":
    demo.launch()