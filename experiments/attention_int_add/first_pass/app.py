import gradio as gr
import numpy as np
import json
from pathlib import Path

from agentic.experiments import benchmark_panel, results_dir, load_task

# Goal directory for benchmark panel
GOAL_DIR = Path(__file__).parent.parent

# Load task for constants
task = load_task(__file__)

VOCAB_SIZE = 15
SEQ_LEN = 14
MAX_DIGITS = 3
SUM_DIGITS = 4
SUM_START_IDX = 9
SUM_POSITIONS = list(range(SUM_START_IDX, SUM_START_IDX + SUM_DIGITS))

PLUS_TOKEN = 10
EQUALS_TOKEN = 11
BOS_TOKEN = 12
EOS_TOKEN = 13
PAD_TOKEN = 14

ID_TO_TOKEN = {i: str(i) for i in range(10)}
ID_TO_TOKEN.update({
    PLUS_TOKEN: "+",
    EQUALS_TOKEN: "=",
    BOS_TOKEN: "<BOS>",
    EOS_TOKEN: "<EOS>",
    PAD_TOKEN: "<PAD>",
})

def decode_seq(ids):
    return " ".join(ID_TO_TOKEN.get(int(i), f"<{i}>") for i in ids)

def find_latest_run():
    results_base = results_dir(__file__).parent
    if not results_base.exists():
        return None
    runs = sorted(results_base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        if (r / "benchmark.json").exists():
            return r
    return None

def load_run(run_dir):
    with open(run_dir / "benchmark.json") as f:
        bench = json.load(f)
    # Also load model for demo predictions
    model_path = run_dir / "model.pt"
    return bench, model_path

def create_addition_problem(a: int, b: int) -> np.ndarray:
    """Create a single input sequence for a + b."""
    a_digits = [(a // 100) % 10, (a // 10) % 10, a % 10]
    b_digits = [(b // 100) % 10, (b // 10) % 10, b % 10]
    
    seq = np.full(SEQ_LEN, PAD_TOKEN, dtype=np.int32)
    seq[0] = BOS_TOKEN
    seq[1:4] = a_digits
    seq[4] = PLUS_TOKEN
    seq[5:8] = b_digits
    seq[8] = EQUALS_TOKEN
    # SUM positions (9-12) stay as PAD
    seq[13] = EOS_TOKEN
    return seq.reshape(1, SEQ_LEN)

def predict_sum(model_fn, a: int, b: int) -> tuple:
    """Run model on a single problem and return predicted sum digits and correctness."""
    input_ids = create_addition_problem(a, b)
    logits = model_fn(input_ids)
    pred_digits = np.argmax(logits[0, SUM_POSITIONS, :], axis=-1)
    true_sum = a + b
    true_digits = [(true_sum // 1000) % 10, (true_sum // 100) % 10, (true_sum // 10) % 10, true_sum % 10]
    correct = np.array_equal(pred_digits, true_digits)
    return pred_digits, true_digits, correct

def make_model_fn_from_checkpoint(model_path):
    """Load a trained model and return model_fn."""
    import torch
    from main import AdditionTransformer, DEVICE
    
    model = AdditionTransformer()
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    
    def model_fn(input_ids: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(input_ids).long().to(DEVICE)
            logits = model(x)
            return logits.detach().cpu().numpy().astype(np.float64)
    
    return model_fn

# Global cache for loaded model
_model_cache = {"model_fn": None, "model_path": None}

def get_model_fn(run_dir):
    model_path = run_dir / "model.pt"
    if _model_cache["model_path"] != model_path:
        _model_cache["model_fn"] = make_model_fn_from_checkpoint(model_path)
        _model_cache["model_path"] = model_path
    return _model_cache["model_fn"]

# ----------------------------------------------------------------------
# Gradio App
# ----------------------------------------------------------------------
with gr.Blocks(title="attention_int_add - first_pass") as demo:
    gr.Markdown("# attention_int_add — first_pass")
    gr.Markdown("Trained 3-layer transformer on 3-digit addition with carry propagation.")
    
    with gr.Row():
        with gr.Column(scale=1):
            run_dropdown = gr.Dropdown(
                label="Select Run",
                choices=[],
                value=None,
                interactive=True,
            )
            refresh_btn = gr.Button("Refresh Runs")
            
        with gr.Column(scale=2):
            run_info = gr.Markdown("Select a run to see details.")
    
    with gr.Tabs():
        with gr.TabItem("Demo"):
            gr.Markdown("## Interactive Demo")
            gr.Markdown("Enter two 3-digit numbers (0-999) to test carry propagation.")
            
            with gr.Row():
                a_input = gr.Number(label="Operand A (0-999)", value=0, minimum=0, maximum=999, precision=0)
                b_input = gr.Number(label="Operand B (0-999)", value=0, minimum=0, maximum=999, precision=0)
            
            predict_btn = gr.Button("Predict", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    input_display = gr.Textbox(label="Input Sequence", interactive=False)
                    pred_display = gr.Textbox(label="Predicted Sum Digits", interactive=False)
                    true_display = gr.Textbox(label="True Sum Digits", interactive=False)
                    result_display = gr.Textbox(label="Result", interactive=False)
                
                with gr.Column():
                    carry_info = gr.Textbox(label="Carry Analysis", interactive=False)
            
            gr.Markdown("### Quick Test: Carry Chain Examples")
            with gr.Row():
                test_099_001 = gr.Button("99 + 1 (2 carries)")
                test_099_901 = gr.Button("99 + 901 (3 carries)")
                test_500_500 = gr.Button("500 + 500 (1 carry)")
                test_999_001 = gr.Button("999 + 1 (3 carries)")
        
        with gr.TabItem("Benchmark"):
            benchmark_panel(GOAL_DIR)
    
    # ------------------------------------------------------------------
    # Event handlers (all INSIDE the Blocks context)
    # ------------------------------------------------------------------
    def refresh_runs():
        results_base = results_dir(__file__).parent
        if not results_base.exists():
            return gr.update(choices=[], value=None)
        runs = sorted(results_base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        choices = [r.name for r in runs if (r / "benchmark.json").exists()]
        return gr.update(choices=choices, value=choices[0] if choices else None)
    
    def on_run_select(run_name):
        if not run_name:
            return "Select a run to see details."
        run_dir = results_dir(__file__).parent / run_name
        bench, _ = load_run(run_dir)
        
        # Format metrics nicely
        lines = [f"**Run: {run_name}**", ""]
        lines.append(f"**Canonical (carries=3) Exact Match:** {bench.get('exact_match_canonical', 0):.2%}")
        lines.append(f"**Carry Robustness:** {bench.get('carry_robustness', 0):.2%}")
        lines.append(f"**Lift Over Baseline (carries=3):** {bench.get('lift_over_baseline_canonical', 0):.2%}")
        lines.append(f"**Mean Exact Match:** {bench.get('exact_match_mean', 0):.2%}")
        lines.append("")
        lines.append("**Per-Carry Slice:**")
        for k in [0, 1, 2, 3]:
            em = bench.get(f'exact_match_carries_{k}', 0)
            base = bench.get(f'linear_baseline_exact_match_carries_{k}', 0)
            lines.append(f"  Carries={k}: EM={em:.2%} | Baseline={base:.2%}")
        
        return "\n".join(lines)
    
    def on_predict(a, b, run_name):
        if not run_name:
            return "", "", "", "Please select a run first.", ""
        run_dir = results_dir(__file__).parent / run_name
        model_fn = get_model_fn(run_dir)
        
        a, b = int(a), int(b)
        pred_digits, true_digits, correct = predict_sum(model_fn, a, b)
        
        # Decode input
        input_ids = create_addition_problem(a, b)
        input_str = decode_seq(input_ids[0])
        
        pred_str = " ".join(str(d) for d in pred_digits)
        true_str = " ".join(str(d) for d in true_digits)
        result_str = "✅ CORRECT" if correct else "❌ INCORRECT"
        
        # Carry analysis
        def count_carries(a, b):
            carry = 0
            n_carries = 0
            for col in range(3):
                da = (a // (10 ** col)) % 10
                db = (b // (10 ** col)) % 10
                total = da + db + carry
                carry = 1 if total >= 10 else 0
                n_carries += carry
            return n_carries
        
        n_carries = count_carries(a, b)
        carry_str = f"This problem has **{n_carries} carry{'s' if n_carries != 1 else ''}**.\n"
        carry_str += f"True sum: {a + b}\n"
        carry_str += f"Predicted: {pred_digits[0]*1000 + pred_digits[1]*100 + pred_digits[2]*10 + pred_digits[3]}"
        
        return input_str, pred_str, true_str, result_str, carry_str
    
    # Wire up events
    refresh_btn.click(refresh_runs, outputs=run_dropdown)
    demo.load(refresh_runs, outputs=run_dropdown)
    
    run_dropdown.change(on_run_select, inputs=run_dropdown, outputs=run_info)
    
    predict_btn.click(
        on_predict,
        inputs=[a_input, b_input, run_dropdown],
        outputs=[input_display, pred_display, true_display, result_display, carry_info]
    )
    
    # Quick test buttons
    for btn, (a, b) in [
        (test_099_001, (99, 1)),
        (test_099_901, (99, 901)),
        (test_500_500, (500, 500)),
        (test_999_001, (999, 1)),
    ]:
        btn.click(
            lambda a, b, r: on_predict(a, b, r),
            inputs=[gr.State(a), gr.State(b), run_dropdown],
            outputs=[input_display, pred_display, true_display, result_display, carry_info]
        )

if __name__ == "__main__":
    demo.launch()