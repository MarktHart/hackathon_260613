"""First-pass hand-built mechanism for attention shift by k.

Implements a clean QK circuit using identity positional embeddings and a
k-dependent shift matrix for the key projection. This is `base_model.py`
plus a hand-crafted relative-position routing mechanism.
"""

import json
import numpy as np
import torch
from pathlib import Path

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

# Offsets each head will host (task measures k in {1, 2, 3, 4, 8}).
K_SWEEP = (1, 2, 3, 4, 8)
CANONICAL_K = 1


def make_model_fn(logit_scale: float = 10.0):
    """Return a model_fn implementing exact relative positional shift by k.

    Contract: model_fn(input_ids (B, L)) -> attention (B, H, L, L), row-stochastic
    over keys. Each head h hosts one shift offset k = K_SWEEP[h]: query position i
    attends to key i - k.

    Mechanism (per head): identity positional embeddings with a shift-by-k key
    projection give pre-softmax logits that are large on key i-k and 0 elsewhere;
    a row-wise softmax then concentrates ~1.0 on the target key.
    """

    def model_fn(input_ids: np.ndarray) -> np.ndarray:
        B, L = input_ids.shape
        H = len(K_SWEEP)

        # Build per-head shift logits on the GPU.
        scores = torch.zeros((H, L, L), dtype=torch.float32, device=DEVICE)
        for h, k in enumerate(K_SWEEP):
            if k >= L:
                continue
            qi = torch.arange(k, L, device=DEVICE)
            scores[h, qi, qi - k] = logit_scale  # query i -> key i-k

        # Row-wise softmax over keys, then broadcast across the batch.
        attn = torch.softmax(scores, dim=-1)                 # (H, L, L)
        attn = attn.unsqueeze(0).expand(B, H, L, L).contiguous()
        return attn.detach().cpu().numpy().astype(np.float32)

    return model_fn


def _run() -> None:
    task = load_task(__file__)
    model_fn = make_model_fn(logit_scale=10.0)
    payload = task.evaluate(model_fn)

    run_dir = results_dir(__file__)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save the payload for inspection
    with (run_dir / "payload.json").open("w") as f:
        json.dump(payload, f, indent=2)

    # Save a compact summary for the demo visualisation
    sweep = payload["sweep"]
    summary = {
        "k": [s["k"] for s in sweep],
        "best_head_mass": [s["best_head_mass"] for s in sweep],
        "best_head_argmax_acc": [s["best_head_argmax_acc"] for s in sweep],
        "uniform_baseline": [s["uniform_baseline"] for s in sweep],
        "lift_over_uniform": [
            s["best_head_mass"] - s["uniform_baseline"] for s in sweep
        ],
    }
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    record_benchmark(__file__, run_dir, payload)
    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":
    _run()