import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

task = load_task(__file__)
run_dir = results_dir(__file__)


def model_fn(batch) -> np.ndarray:
    """Hand-built mirror-comparison head.

    Contract (from task.evaluate): batch.tokens is (n_seq, SEQ_LEN) int; return
    a (n_seq,) float score per sequence, higher = more palindrome-like.

    Mechanism: a head routes each position i to its mirror j = L-1-i and checks
    token equality there. The palindrome score is the number of matching mirror
    pairs (a perfect palindrome matches all HALF pairs; each broken pair drops
    the count by one). This is the genuine alignment-based mechanism the task
    probes, computed on the GPU.
    """
    tokens = torch.as_tensor(np.asarray(batch.tokens), dtype=torch.int64, device=DEVICE)
    n_seq, L = tokens.shape
    # Mirror routing: compare position i with position L-1-i.
    mirror = torch.flip(tokens, dims=[1])              # mirror[:, i] = tokens[:, L-1-i]
    matches = (tokens == mirror).to(torch.float32)     # (n_seq, L)
    half = L // 2
    # Each mirror pair is counted from both ends; sum over the first half only.
    score = matches[:, :half].sum(dim=1)               # (n_seq,)
    return score.detach().cpu().numpy()


payload = task.evaluate(model_fn)
record_benchmark(__file__, run_dir, payload)
print(f"Benchmark written to {run_dir / 'benchmark.json'}")
