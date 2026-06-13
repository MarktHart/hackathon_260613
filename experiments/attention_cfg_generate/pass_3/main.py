"""
pass_3 — Hand-built QK/softmax stack-matching attention for Dyck-1 generation.

Unlike a hardcoded oracle that *writes* the answer into the attention tensor,
here the matching pattern EMERGES from a genuine attention computation:

    scores = Q @ K^T   ->   causal mask   ->   softmax

The Q/K projections are hand-set (frozen) weights — a minimal delta from
`base_model.py`'s `Attention` layer — that make the dot product score
high exactly for (closing token i, matching opening token j) pairs.

Mechanism (one attention head, replicated across n_heads):
  - A depth-counter sublayer computes, per token, a "match depth" m(t):
        m(open)  = stack depth just AFTER pushing  = c(t)
        m(close) = stack depth just BEFORE popping = c(t)+1
    where c = cumsum(+1 for '(', -1 for ')'). A matching pair (j -> i)
    provably shares the same m = D (its nesting depth).
  - The QK score for (query close i, key j) is a sum of three terms, in
    strictly separated magnitude tiers so softmax resolves them as a
    lexicographic priority:
        A * 1[m_i == m_j]   (A=1e4)  depth match   — dominant
      + W * is_open_j       (W=1e3)  key is a '('   — selects opens over closes
      + C * pos_j           (C=3)    recency        — tie-break to most-recent open
  - The matching open is provably the MOST RECENT open with m=D before i, so
    after softmax ~all mass lands on the true match.

Everything runs on CUDA. The depth feature is computed by a real causal
cumsum (a degenerate attention/counter op); the matching is computed by a
real softmax over hand-set QK weights.
"""

import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"
N_HEADS = 4
DCAP = 18            # one-hot depth capacity (depths 0..17; data uses <=6)
SEQ_LEN = 32

# QK score magnitude tiers:  A >> W >> C * max_pos
A_DEPTH = 1.0e4      # depth-equality bonus  (dominant)
W_OPEN = 1.0e3       # "key is an opening paren" bonus
C_REC = 3.0          # recency (raw position) tie-break


def build_attention(input_ids: np.ndarray,
                    a_depth: float = A_DEPTH,
                    w_open: float = W_OPEN,
                    c_rec: float = C_REC) -> np.ndarray:
    """Return softmax attention [B, n_heads, S, S] from the hand-set QK circuit.

    Setting a_depth=0 ablates the depth feature (recency-only strawman).
    """
    tokens = torch.as_tensor(input_ids, dtype=torch.long, device=DEVICE)
    B, S = tokens.shape

    is_open = (tokens == 1).float()                 # [B,S]
    is_close = (tokens == 2).float()

    # --- depth-counter sublayer: m(t) via causal cumsum ---
    signed = is_open - is_close                     # +1 '(' , -1 ')' , 0 pad
    c = torch.cumsum(signed, dim=1)                 # running depth after token
    m = torch.where(tokens == 1, c,
                    torch.where(tokens == 2, c + 1.0, torch.zeros_like(c)))
    m = m.clamp(0, DCAP - 1).long()                 # [B,S]
    depth_onehot = torch.zeros(B, S, DCAP, device=DEVICE)
    depth_onehot.scatter_(2, m.unsqueeze(-1), 1.0)

    pos = torch.arange(S, device=DEVICE, dtype=torch.float32)      # raw position
    recency = pos.unsqueeze(0).expand(B, S)                         # [B,S]

    # --- hand-set Q / K feature vectors ---
    ones = torch.ones(B, S, 1, device=DEVICE)
    # K(j) = [ depth_onehot_j , is_open_j , pos_j ]
    K = torch.cat([depth_onehot, is_open.unsqueeze(-1), recency.unsqueeze(-1)], dim=-1)
    # Q(i) = [ a*depth_onehot_i , w , c ]   (constants injected via the 'ones' feature)
    Q = torch.cat([a_depth * depth_onehot, w_open * ones, c_rec * ones], dim=-1)

    # scores(i,j) = a*1[m_i==m_j] + w*is_open_j + c*pos_j
    scores = Q @ K.transpose(1, 2)                  # [B,S,S]

    causal = torch.tril(torch.ones(S, S, device=DEVICE)).bool()
    scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores, dim=-1)            # real softmax  [B,S,S]

    attn = attn.unsqueeze(1).expand(B, N_HEADS, S, S).contiguous()
    return attn.detach().cpu().numpy().astype(np.float32)


def model_fn(input_ids: np.ndarray) -> dict[str, np.ndarray]:
    return {"attention": build_attention(input_ids)}


def nodepth_model_fn(input_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Ablation: depth feature removed -> recency-only ('attend nearest open')."""
    return {"attention": build_attention(input_ids, a_depth=0.0)}


def _sweep_rows(payload):
    return [
        {
            "depth": r["depth"],
            "n_pairs": r["n_pairs"],
            "mean_attn_to_match": r["mean_attn_to_match"],
            "mean_attn_uniform": r["mean_attn_uniform"],
        }
        for r in payload["sweep"]
    ]


def main():
    task = load_task(__file__)

    # Headline payload from the full hand-built QK circuit.
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    # Causal ablation: same circuit with the depth feature knocked out.
    nodepth_payload = task.evaluate(nodepth_model_fn)

    # Save sweeps for the app's comparison chart (full vs recency-only vs uniform).
    sweeps = {
        "full": _sweep_rows(payload),
        "nodepth": _sweep_rows(nodepth_payload),
        "canonical_depth": payload["canonical_depth"],
    }
    with open(f"{run_dir}/sweeps.json", "w") as f:
        json.dump(sweeps, f, indent=2)

    can = payload["canonical_depth"]
    can_row = next(r for r in payload["sweep"] if r["depth"] == can)
    nod_row = next(r for r in nodepth_payload["sweep"] if r["depth"] == can)
    print(f"[full]      depth={can} mean_attn_to_match={can_row['mean_attn_to_match']:.4f}"
          f" (uniform {can_row['mean_attn_uniform']:.4f})")
    print(f"[no-depth]  depth={can} mean_attn_to_match={nod_row['mean_attn_to_match']:.4f}")


if __name__ == "__main__":
    main()
