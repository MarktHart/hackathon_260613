"""attention_palindrome / pass_3 — hand-built mirror-comparison attention head.

Approach (hand_built): the smallest possible delta from `base_model.py`'s
single attention head, with every weight set BY HAND (no training). The head
implements exactly the mechanism the goal describes:

  * Q/K depend only on *position*, wired so query position i attends to key
    position L-1-i (the mirror). Softmax over a one-hot dot product with a high
    temperature makes this a hard i -> L-1-i routing (the anti-diagonal).
  * V carries the *token identity* (one-hot). So the attention output at
    position i is the one-hot of the token sitting at the mirror position.
  * A dot product of the token one-hot at i with the attended mirror one-hot
    yields 1 iff token_i == token_{L-1-i}. Summed over positions, the score is
    the number of agreeing positions = SEQ_LEN - 2k for a k-broken negative.

Higher score = more palindrome-like, monotone in k, so the rank-AUC is a clean
1.0 at every slice — including the diagnostic k=1 anchor where a histogram
readout is at chance.

Everything runs as torch tensors on CUDA. We also evaluate two *ablations*
(identity routing, off-by-one routing) to show the mechanism is load-bearing:
remove the mirror routing and the separation collapses to chance.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # hard requirement: real compute on the GPU, no CPU fallback

task = load_task(__file__)
run_dir = results_dir(__file__)

SEQ_LEN = int(task.SEQ_LEN)   # 16
VOCAB = int(task.VOCAB)       # 8
TEMP = 30.0                   # softmax sharpness for the positional routing


def attention_pattern(routing: str = "mirror", temp: float = TEMP) -> torch.Tensor:
    """Hand-set positional attention matrix A (L, L), depends only on position.

    K position j -> one-hot(j). Q position i -> one-hot(target(i)). The dot
    product is `temp` exactly when j == target(i) and 0 otherwise, so softmax
    routes (near) all mass to the target. routing chooses the target map:
      * 'mirror'   : target(i) = L-1-i   (the correct palindrome mechanism)
      * 'identity' : target(i) = i       (ablation: compare a token with itself)
      * 'shift'    : target(i) = L-2-i   (ablation: off-by-one mirror)
    """
    L = SEQ_LEN
    idx = torch.arange(L, device=DEVICE)
    if routing == "mirror":
        target = (L - 1) - idx
    elif routing == "identity":
        target = idx
    elif routing == "shift":
        target = ((L - 2) - idx) % L
    else:
        raise ValueError(f"unknown routing {routing!r}")
    K_pos = torch.eye(L, device=DEVICE)                 # (L, L) one-hot keys
    Q_pos = torch.eye(L, device=DEVICE)[target]         # (L, L) one-hot queries
    pos_scores = (Q_pos @ K_pos.t()) * temp             # (L, L)
    return torch.softmax(pos_scores, dim=-1)            # (L, L)


def circuit_scores(tokens_np: np.ndarray, routing: str = "mirror") -> np.ndarray:
    """Run the hand-built head on CUDA and return per-sequence palindrome score."""
    tokens = torch.as_tensor(tokens_np, dtype=torch.long, device=DEVICE)
    E = torch.nn.functional.one_hot(tokens, num_classes=VOCAB).float()  # (B,L,V)
    A = attention_pattern(routing)                                      # (L,L)
    # attention output at i = one-hot of the token at position target(i)
    O = torch.einsum("ij,bjv->biv", A, E)                              # (B,L,V)
    match = (E * O).sum(dim=-1)                                         # (B,L) ~1 if equal
    return match.sum(dim=-1).detach().cpu().numpy()                    # (B,)


def model_fn(batch) -> np.ndarray:
    return circuit_scores(batch.tokens, routing="mirror")


# ---- headline run: the real mechanism --------------------------------------
payload = task.evaluate(model_fn)
record_benchmark(__file__, run_dir, payload)

# ---- ablations + artefacts for the visualisation ---------------------------
# Re-evaluate under each ablated routing to show the mirror routing is the part
# that matters (these are NOT recorded as benchmarks — they feed the demo).
ablations = {}
for name in ("identity", "shift"):
    abl_payload = task.evaluate(lambda b, r=name: circuit_scores(b.tokens, routing=r))
    ablations[name] = abl_payload["sweep"]

artifacts = {
    "seq_len": SEQ_LEN,
    "vocab": VOCAB,
    "temp": TEMP,
    "mismatch_sweep": list(task.MISMATCH_SWEEP),
    "attention_pattern": attention_pattern("mirror").detach().cpu().numpy().tolist(),
    "sweep_model": payload["sweep"],
    "sweep_baseline": payload["linear_baseline"],
    "sweep_ablation": ablations,
}
with open(run_dir / "artifacts.json", "w") as f:
    json.dump(artifacts, f, indent=2)

# quick console summary
print("mirror routing AUC per slice:",
      {r["mismatch"]: round(r["auc"], 3) for r in payload["sweep"]})
print("histogram baseline   per slice:",
      {r["mismatch"]: round(r["auc"], 3) for r in payload["linear_baseline"]})
print(f"Benchmark + artifacts written to {run_dir}")
