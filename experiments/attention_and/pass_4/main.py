"""attention_and / pass_4 — hand-built ReLU-gated bilinear AND head.

Mechanism (a small delta from base_model.py's attention score):
    p_A = relu(residual @ q_A)      # query-feature A read-off, negatives clipped
    p_B = relu(residual @ q_B)      # query-feature B read-off
    logit = beta * p_A * p_B        # multiplicative AND: nonzero only if BOTH fire

This is the minimal nonlinear conjunction. A *linear* head (q_A+q_B, the task's
baseline) lights up for single features too; the product gates them off because
either factor being ~0 zeroes the logit. ReLU removes negative-noise leakage so
"neither" positions stay near zero instead of producing a spurious positive
product. All compute runs in torch on CUDA.
"""
import numpy as np
import torch

DEVICE = "cuda"  # pipeline guarantees a visible GPU; no CPU fallback
BETA = 2.0       # temperature: sharpens the post-softmax AND peak


def model_fn(q_A: np.ndarray, q_B: np.ndarray, residual: np.ndarray) -> np.ndarray:
    q_At = torch.as_tensor(q_A, dtype=torch.float32, device=DEVICE)
    q_Bt = torch.as_tensor(q_B, dtype=torch.float32, device=DEVICE)
    rt = torch.as_tensor(residual, dtype=torch.float32, device=DEVICE)

    p_A = torch.relu(rt @ q_At)              # (n_positions,)
    p_B = torch.relu(rt @ q_Bt)              # (n_positions,)
    logits = BETA * p_A * p_B                # multiplicative AND
    return logits.detach().cpu().numpy()


if __name__ == "__main__":
    from agentic.experiments import load_task, record_benchmark, results_dir

    task = load_task(__file__)
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    print("and_sharpness sweep:",
          [(r["cosine"], round(r["and_sharpness"], 3)) for r in payload["sweep"]])
