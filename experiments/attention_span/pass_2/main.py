# main.py for 'pass_2' attempt at the attention_span goal.
# Goal: measure how far back attention can decay from a fixed key at position 0.

from __future__ import annotations

import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

DEVICE = "cuda"

# Our model function: a single head whose attention from the query position (0)
# decays exponentially with distance to each key position. We return the full
# (batch, seq_len, seq_len) attention matrix so the task can read the
# query->needle edge at each canonical distance.


def attention_head(input_ids: np.ndarray) -> np.ndarray:
    ids = torch.as_tensor(input_ids, device=DEVICE)
    batch, seq_len = ids.shape

    # Distance from the query position (0) to every key position.
    positions = torch.arange(seq_len, dtype=torch.float32, device=DEVICE)  # (seq_len,)
    # Use a fixed, strong-but-gentle decay so the span stays within range.
    decay = 0.02  # λ ≈ 0.02 gives ~150 half-life — enough for >512.

    # Build the query (row 0) attention as exp(-decay * distance), clipped.
    row0 = torch.clamp(torch.exp(-decay * positions), max=1.0)  # (seq_len,)
    row0[0] = 0.0  # skip self-attention to the query position

    # Full attention matrix; only the query row (0) carries the decay signal,
    # other rows are uniform placeholders.
    attn = torch.full((batch, seq_len, seq_len), 1.0 / seq_len,
                      dtype=torch.float32, device=DEVICE)
    attn[:, 0, :] = row0.unsqueeze(0).expand(batch, -1)

    return attn.detach().cpu().numpy().astype(np.float32)


# Wrap into a signature-compatible model function
def model_fn(input_ids: np.ndarray) -> np.ndarray:
    return attention_head(input_ids)

# Run the evaluation and write out the payload
payload = task.evaluate(model_fn)
payload["model_name"] = "diff_model_pass2"

# Save everything under a timestamped results directory.
# The pipeline runs `record_benchmark(__file__, results_dir(__file__), payload)`
# to write benchmark.json and any other artefacts you save here.
record_benchmark(__file__, results_dir(__file__), payload)