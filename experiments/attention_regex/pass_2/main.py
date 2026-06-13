"""attention_regex / pass_2

Hand-built MULTI-HEAD ATTENTION circuit for regex-like pattern matching.

Delta from `base_model.py`: a single multi-head self-attention block whose
weights are set by hand (no training) and whose head-readout is changed from
the standard concat+project to an element-wise MIN across heads.

  * One head per CONCRETE (non-wildcard) pattern offset j.
  * Head j uses a sharp *relative-position* bias so its softmax collapses to a
    one-hot that gathers exactly the neighbour t = i-(L-1-j) for query i.
    (This is real softmax attention over an (N,N) score matrix, run on CUDA.)
  * The value each head reads is the token-match score
        sim_j(t) = residual[t] . embed[pattern[j]]
    i.e. how well position t looks like the token the pattern requires there.
  * Readout = MIN over heads.  min == logical AND in score space: a window is
    a match-end only if EVERY concrete offset matches.  Summing (the previous
    attempt) is OR-ish and leaks false positives as L grows; MIN does not.

The whole thing is analytic — weights derived from `pattern` + `embed`.
"""

import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # guaranteed by the pipeline; no CPU fallback

# Hand-set hyperparameters of the circuit.
POS_BIAS = 30.0   # sharpness of the relative-position selection (-> one-hot gather)
BETA = 2.0        # output temperature; sharpens the downstream softmax over positions


def model_fn(pattern: np.ndarray, embed: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Multi-head attention regex matcher (see module docstring).

    Args:
        pattern:  (L,) int ids in [0, V); -1 == wildcard.
        embed:    (V, d) unit-norm token embeddings.
        residual: (N, d) embedded sequence + noise.
    Returns:
        (N,) per-position logits; high where the pattern finishes matching.
    """
    pattern_t = torch.as_tensor(pattern, dtype=torch.long, device=DEVICE)
    embed_t = torch.as_tensor(embed, dtype=torch.float32, device=DEVICE)
    residual_t = torch.as_tensor(residual, dtype=torch.float32, device=DEVICE)

    N, d = residual_t.shape
    L = int(pattern_t.shape[0])

    concrete = torch.where(pattern_t >= 0)[0]  # (K,) offsets that constrain the match
    if concrete.numel() == 0:
        # All wildcards (generator forbids this, but stay safe): no constraint.
        return torch.zeros(N, dtype=torch.float32, device=DEVICE).detach().cpu().numpy()

    pos = torch.arange(N, device=DEVICE)
    rel = pos[:, None] - pos[None, :]          # rel[i,t] = i - t   (relative position)

    head_logits = []
    for j in concrete.tolist():
        shift = L - 1 - j                       # head j gathers neighbour t = i - shift
        target = embed_t[pattern_t[j]]          # (d,) the token this offset requires
        sim = residual_t @ target               # (N,) value at every key position

        # Real softmax attention with a relative-position bias peaked at rel==shift.
        bias = -POS_BIAS * (rel.float() - float(shift)).abs()   # (N,N)
        attn = torch.softmax(bias, dim=1)       # (N,N) ~ one-hot at t = i-shift
        head_logits.append(attn @ sim)          # (N,) == sim[i-shift]

    H = torch.stack(head_logits, dim=0)         # (K, N)
    logit = H.min(dim=0).values                 # AND across concrete offsets

    # Windows that don't fit (i < L-1) can never be a match-end.
    logit = logit.masked_fill(pos < (L - 1), -1e9)

    logit = BETA * logit
    return logit.detach().cpu().numpy()


def main():
    task = load_task(__file__)
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    # Also persist the raw payload so the app's sweep panel can plot per-run.
    (run_dir / "payload.json").write_text(json.dumps(payload, indent=2))

    print(f"Benchmark recorded to {run_dir / 'benchmark.json'}")
    canon = next(r for r in payload["sweep"] if r["length"] == task.CANONICAL_LENGTH)
    print(f"canonical (L=3) match_sharpness: {canon['match_sharpness']:.3f}")


if __name__ == "__main__":
    main()
