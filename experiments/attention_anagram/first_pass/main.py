"""attention_anagram / first_pass — hand-built token-matching attention head.

Hypothesis: a *single* attention head can solve the anagram-alignment task with
no learning at all, by setting the query/key circuit to be token-identity
matching. Since the target sequence is a permutation of the source, the target
token at position t equals the source token at its true source position. If the
query for a target token is its one-hot token vector and the key for a source
token is *its* one-hot token vector, then QK^T scores are 1 exactly where the
tokens match and 0 elsewhere. A sharp softmax over source positions therefore
concentrates attention on the matching source position — i.e. on the true
permutation pre-image.

This is `base_model.py` reduced to a single attention layer (no MLP), with
W_Q = W_K = identity acting on a one-hot token embedding and a temperature on
the logits. Everything runs as torch tensors on CUDA.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU; no CPU fallback.

VOCAB_SIZE = 50
N_HEADS = 8


def make_model_fn(temperature: float = 30.0):
    """Build the hand-set token-matching model_fn.

    The 8 heads share the identity QK circuit; we give them a *spread* of
    temperatures so the demo can show how logit sharpness controls alignment,
    while keeping every head sharp enough to dominate the uniform baseline.
    """
    # Temperatures per head: all sharp, mild spread (sharpest first).
    temps = torch.linspace(temperature, temperature * 0.6, N_HEADS, device=DEVICE)

    def model_fn(src_ids: np.ndarray, tgt_ids: np.ndarray) -> np.ndarray:
        src = torch.as_tensor(src_ids, dtype=torch.long, device=DEVICE)
        tgt = torch.as_tensor(tgt_ids, dtype=torch.long, device=DEVICE)
        B, L = src.shape

        # One-hot token embedding == identity W_Q / W_K. (B, L, V)
        src_oh = F.one_hot(src, VOCAB_SIZE).float()
        tgt_oh = F.one_hot(tgt, VOCAB_SIZE).float()

        # QK^T: match[b, t, s] = 1 iff tgt token t == src token s. (B, L, L)
        match = torch.einsum("btv,bsv->bts", tgt_oh, src_oh)

        # Per-head temperature scaling of the logits, then softmax over source.
        scores = match.unsqueeze(1) * temps.view(1, N_HEADS, 1, 1)  # (B, H, L, L)
        attn = torch.softmax(scores, dim=-1)
        return attn.detach().cpu().numpy().astype(np.float32)

    return model_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=30.0,
                    help="Base logit temperature for the matching head.")
    args = ap.parse_args()

    task = load_task(__file__)

    model_fn = make_model_fn(args.temperature)
    payload = task.evaluate(model_fn)

    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    # Save the full payload for the Gradio Demo tab to read back.
    with open(run_dir / "payload.json", "w") as f:
        json.dump(payload, f, indent=2)

    # Console summary — load the goal's benchmark.py by path.
    import importlib.util
    from pathlib import Path
    bpath = Path(__file__).resolve().parent.parent / "benchmark.py"
    spec = importlib.util.spec_from_file_location("anagram_benchmark", bpath)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    metrics = bench.score(payload)
    print("=== metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
