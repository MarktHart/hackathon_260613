"""
Hand-built attention head for k-th position selection.

Attempt type: hand_built (no training).

The model_fn signature (from task.py) is:
    model_fn(input_ids: (B, L) int32, positions: (L,) int32) -> (B, L) float32 attn

Crucially, model_fn is NOT told `k`. The only signal of `k` is statistical:
position `k` holds the marker token M=99 in *every* sequence of the batch,
whereas every other position holds it only with probability ~1/V by chance.
So a content-matching head (attend to value 99) splits its mass onto spurious
markers, but a head that addresses by *position* can isolate k cleanly.

The hand-built circuit:
  1. marker_mask = (input_ids == 99)                      # (B, L)
  2. pos_freq = marker_mask.mean(over batch)              # (L,)  -> ~1.0 at k, ~0.01 elsewhere
  3. k_hat = argmax(pos_freq)                             # detect the addressed position
  4. keys  = identity position embeddings P (L x L)       # key for pos l is one-hot(l)
     query = BETA * one_hot(k_hat)                        # a positional query selecting k_hat
     scores = query @ keys^T = BETA * one_hot(k_hat)      # (B, L), same query for every row
     attn   = softmax(scores)                             # near-delta spike at k_hat

This is `base_model.py` reduced to a single attention head with hand-set,
position-only Q/K weights (no MLP, no value projection needed for the metric).
"""

import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU; no CPU fallback.

task = load_task(__file__)

MARKER = 99
BETA = 25.0  # inverse-temperature of the positional query -> softmax sharpness


def positional_head_fn(input_ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Hand-built positional k-th selection head. All compute on CUDA."""
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=DEVICE)      # (B, L)
    _ = torch.as_tensor(positions, dtype=torch.long, device=DEVICE)        # (L,)
    B, L = ids.shape

    # Step 1-3: detect the position addressed across the batch.
    marker_mask = (ids == MARKER).float()                                  # (B, L)
    pos_freq = marker_mask.mean(dim=0)                                     # (L,)
    k_hat = torch.argmax(pos_freq)                                         # scalar

    # Step 4: positional QK attention. Keys = identity (one-hot per position),
    # query = BETA * one_hot(k_hat). scores = query @ keys^T.
    keys = torch.eye(L, device=DEVICE, dtype=torch.float32)                # (L, L)
    query = BETA * keys[k_hat]                                             # (L,)
    scores = (query.unsqueeze(0) @ keys.t()).expand(B, -1)                 # (B, L)
    attn = torch.softmax(scores, dim=1)                                    # (B, L)
    return attn.detach().cpu().numpy().astype(np.float32)


def content_matching_fn(input_ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Strawman: attend uniformly to every position whose token == marker 99.

    Fails to isolate k whenever a spurious marker appears elsewhere.
    Runs on CUDA too (kept off the official payload; used only for the viz).
    """
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=DEVICE)
    mask = (ids == MARKER).float()                                         # (B, L)
    attn = mask / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return attn.detach().cpu().numpy().astype(np.float32)


def _mean_attn_vectors(model_fn) -> list[dict]:
    """Per-k mean attention distribution over positions, for visualisation."""
    records = []
    for batch in task.generate(seed=0):
        attn = model_fn(batch.input_ids, batch.positions)                 # (B, L)
        k = int(batch.target_k)
        records.append({
            "k": k,
            "mean_attn": attn.mean(axis=0).astype(float).tolist(),         # (L,)
            "attn_at_k": float(attn[:, k].mean()),
            "attn_max_pos": float(np.mean(np.argmax(attn, axis=1))),
        })
    return records


def main() -> None:
    run_dir = results_dir(__file__)

    # Official benchmark payload comes from the hand-built positional head.
    payload = task.evaluate(positional_head_fn)
    payload["model_name"] = "handbuilt-positional-kth-head"

    # Visualisation artefact: compare positional head vs content strawman vs uniform.
    uniform_fn = task.random_model_fn()
    comparison = {
        "k_list": [int(b.target_k) for b in task.generate(seed=0)],
        "L": 32,
        "beta": BETA,
        "methods": {
            "positional": _mean_attn_vectors(positional_head_fn),
            "content":    _mean_attn_vectors(content_matching_fn),
            "uniform":    _mean_attn_vectors(uniform_fn),
        },
    }
    with open(run_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    record_benchmark(__file__, run_dir, payload)
    print(f"wrote benchmark + comparison to {run_dir}")
    print("attn_at_k (positional):",
          [round(r["attn_at_k"], 4) for r in comparison["methods"]["positional"]])


if __name__ == "__main__":
    main()
