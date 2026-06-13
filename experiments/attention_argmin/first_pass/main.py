import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

# This file runs the experiment once.
# It imports the ground-truth data and evaluator from the goal's task.py.
task = load_task(__file__)

# The function your attempt contributes.
def attention_argmin_model_fn(keys: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """
    Attention-style model that treats each value as an embed (value = Q)
    and computes self-attention over the sequence, with the score between
    key and query defined by negative value (K, V also = -values).
    Lower values get larger weights because softmax(-value) is high when value is small.

    task.evaluate calls model_fn per sequence with
    keys (seq_len, key_dim), values (seq_len,), query (key_dim,) and expects
    an attention distribution of shape (seq_len,).
    """
    vt = torch.as_tensor(values, dtype=torch.float32, device=DEVICE)  # (seq_len,)

    # logits derived from -values → lower value → higher logit.
    # This approximates argmin: softmax(-x) emphasizes the minimum.
    logits = -vt
    attn = torch.softmax(logits, dim=-1)   # (seq_len,)

    # Return as a valid attention distribution
    return attn.detach().cpu().numpy()

# ----- end of model contribution ---------------------------------------------------------

# Now drive the canonical sweep from task.py.
payload = task.evaluate(attention_argmin_model_fn)

# Write results under results/<timestamp>/...
run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)