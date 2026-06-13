"""attention_brackets / pass_3 — hand-built stack-matching attention head.

A *closing* bracket's query is routed to its *matching opener* (the position a
parser's stack would pop) using a genuine QK attention circuit — not an oracle
lookup of the match array.

Mechanism (a small delta from `experiments/base_model.py` self-attention):

  1. A structural feature `level[i]` = the stack height that token `i` sits at.
     This is a running signed cumsum of the token stream (+1 per `(`, then read;
     read-then -1 per `)`), exactly what one extra causal attention head over a
     signed token embedding would compute. No match array is used.
  2. Query (closers) and Key (openers) are one-hot encodings of `level`. Their
     dot product is 1 iff a closer and an opener share the same nesting level —
     the stack-matching condition.
  3. A tiny recency bias `ALPHA * position` breaks ties toward the most-recent
     same-level opener — which is precisely the one the stack pops.

So `score[i,j] = C * [level_i == level_j and j opens and i closes] + ALPHA*j`,
masked to opener keys and causal. With `C >> ALPHA * L`, a same-level opener
always outscores any non-matching key, and recency selects the true match.
This is the opposite of pass_2's broken scalar `q·k` (which peaked on the last
position, not the matching opener).
"""

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

DEVICE = "cuda"  # pipeline guarantees a visible GPU; no CPU fallback

OPEN, CLOSE, PAD = 0, 1, 2
C = 300.0       # weight on the level-match term (>> ALPHA * SEQ_LEN)
ALPHA = 4.0     # recency tiebreak among same-level openers


def _levels(tokens: np.ndarray):
    """Running stack height per token plus open/close role masks.

    level[i] = height of the slot token i occupies. Computed purely from the
    token stream (a signed running sum) — never from the ground-truth match."""
    L = len(tokens)
    level = np.zeros(L, dtype=np.int64)
    is_open = np.zeros(L, dtype=bool)
    is_close = np.zeros(L, dtype=bool)
    h = 0
    for i, t in enumerate(tokens):
        if t == OPEN:
            h += 1
            level[i] = h          # opener occupies the slot it just pushed
            is_open[i] = True
        elif t == CLOSE:
            level[i] = h          # closer pops the current top slot
            is_close[i] = True
            h = max(h - 1, 0)
        # PAD: level 0, no role
    return level, is_open, is_close


def stack_match_model_fn(tokens: np.ndarray) -> np.ndarray:
    """model_fn contract: tokens (L,) int -> attention (L, L) float, on GPU."""
    tokens = np.asarray(tokens).astype(np.int64)
    L = int(tokens.shape[0])
    level, is_open, is_close = _levels(tokens)
    D = L + 2  # one-hot width (levels never exceed L/2)

    lvl = torch.as_tensor(level, device=DEVICE)
    onehot = torch.zeros((L, D), device=DEVICE, dtype=torch.float32)
    onehot[torch.arange(L, device=DEVICE), lvl] = 1.0

    opent = torch.as_tensor(is_open, device=DEVICE, dtype=torch.float32)
    closet = torch.as_tensor(is_close, device=DEVICE, dtype=torch.float32)

    Q = onehot * closet[:, None]            # queries only for closers
    K = onehot * opent[:, None]             # keys only for openers
    same_level = Q @ K.t()                  # (L, L): 1 iff shared nesting level

    pos = torch.arange(L, device=DEVICE, dtype=torch.float32)
    logits = C * same_level + ALPHA * pos[None, :]

    neg = torch.full_like(logits, -1e9)
    logits = torch.where(opent[None, :] > 0, logits, neg)         # opener keys only
    causal = torch.tril(torch.ones((L, L), device=DEVICE))
    logits = torch.where(causal > 0, logits, neg)                 # causal mask

    attn = torch.softmax(logits, dim=1)     # row-stochastic, numerically stable
    return attn.detach().cpu().numpy()


def main():
    payload = task.evaluate(stack_match_model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    print("attention_brackets / pass_3 — stack-match head")
    for rec in payload["sweep"]:
        base = rec["uniform_baseline_mass"]
        lift = (rec["match_mass"] - base) / max(1e-9, 1.0 - base)
        print(
            f"  depth {rec['depth']}: acc={rec['match_accuracy']:.3f} "
            f"mass={rec['match_mass']:.3f} base={base:.3f} lift={lift:.3f}"
        )


if __name__ == "__main__":
    main()
