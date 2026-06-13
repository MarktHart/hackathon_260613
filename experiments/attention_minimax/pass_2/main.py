"""attention_minimax / pass_2 — confidence-gated (minimax) attention head.

Hypothesis
----------
A standard softmax head has *no notion of absolute match quality*: it softmaxes
whatever scores it is handed, so when the only similarities present are
incidental noise it still leaks mass onto the most spuriously-similar key. The
minimax-optimal response when no key is a genuine match is to spread mass
uniformly (max weight = 1/3).

Mechanism (smallest delta from base_model.py attention)
-------------------------------------------------------
We keep scaled dot-product attention and add ONE hand-set gate on the
inverse-temperature:

    s_i  = (k_i · q) / sqrt(d)                 # standard scaled scores
    beta = BETA_MAX * relu(max_i s_i - TAU)    # confidence gate (scalar)
    w    = softmax( beta * (s - max(s)) )       # gated softmax

`TAU` is an absolute "is this a real match?" threshold. When the best score is
below TAU (only incidental similarity present) the gate is exactly 0, the
logits collapse to 0, and the head emits the uniform / minimax distribution.
When a genuinely target-aligned key appears, max score exceeds TAU, the gate
opens, and the head reverts to ordinary concentrating attention.

The same gate is causal both ways — this file also records a "target-injected"
sweep (a real TARGET key spliced into the keys) showing the gate OPEN and the
head concentrate, proving the uniform output is computed, not hard-wired.
"""

import json
import math

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a GPU; never fall back to CPU

# ---- hand-set circuit constants -------------------------------------------
TAU = 0.50       # absolute match-quality threshold (incidental scores ~0.16-0.24)
BETA_MAX = 20.0  # inverse-temperature gain once the gate opens


def gated_attention(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Confidence-gated scaled-dot-product attention. Real compute on CUDA."""
    q = torch.as_tensor(query, dtype=torch.float32, device=DEVICE)
    K = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)
    d = q.shape[-1]
    scores = (K @ q) / math.sqrt(d)                       # (n,)
    max_score = scores.max()
    beta = BETA_MAX * torch.relu(max_score - TAU)         # scalar gate
    logits = beta * (scores - max_score)                  # shift -> stable
    w = torch.softmax(logits, dim=0)
    return w.detach().cpu().numpy().astype(np.float32)


# ---- strawmen (for the demo / baseline comparison) ------------------------
def softmax_scaled(query, keys):
    q = torch.as_tensor(query, dtype=torch.float32, device=DEVICE)
    K = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)
    s = (K @ q) / math.sqrt(q.shape[-1])
    return torch.softmax(s, dim=0).detach().cpu().numpy().astype(np.float32)


def softmax_raw(query, keys):
    q = torch.as_tensor(query, dtype=torch.float32, device=DEVICE)
    K = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)
    s = K @ q  # no 1/sqrt(d) scaling — the naive choice
    return torch.softmax(s, dim=0).detach().cpu().numpy().astype(np.float32)


def linear_attn(query, keys):
    q = torch.as_tensor(query, dtype=torch.float32, device=DEVICE)
    K = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)
    s = K @ q
    s = s - s.min() + 1e-8
    return (s / s.sum()).detach().cpu().numpy().astype(np.float32)


def _gate_value(query, keys):
    q = torch.as_tensor(query, dtype=torch.float32, device=DEVICE)
    K = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)
    s = (K @ q) / math.sqrt(q.shape[-1])
    return float(s.max().item()), float((BETA_MAX * torch.relu(s.max() - TAU)).item())


def _mw(w):
    return float(np.max(w))


def main():
    task = load_task(__file__)
    run_dir = results_dir(__file__)

    # ---- benchmark payload: ONLY the gated head is scored ----------------
    payload = task.evaluate(gated_attention)
    record_benchmark(__file__, run_dir, payload)

    # ---- demo artefacts ---------------------------------------------------
    batches = task.generate(seed=0)
    alphas = [b.alpha for b in batches]

    demo = {"tau": TAU, "beta_max": BETA_MAX, "alphas": alphas,
            "canonical": [], "target_injected": []}

    for b in batches:
        ms, beta = _gate_value(b.query, b.keys)
        g = gated_attention(b.query, b.keys)
        ss = softmax_scaled(b.query, b.keys)
        sr = softmax_raw(b.query, b.keys)
        ln = linear_attn(b.query, b.keys)
        demo["canonical"].append({
            "alpha": b.alpha, "max_score": ms, "beta": beta,
            "gated": g.tolist(), "gated_mw": _mw(g),
            "softmax_scaled": ss.tolist(), "softmax_scaled_mw": _mw(ss),
            "softmax_raw": sr.tolist(), "softmax_raw_mw": _mw(sr),
            "linear": ln.tolist(), "linear_mw": _mw(ln),
        })

    # ---- causal check: splice the real TARGET in as a 4th key ------------
    e_target = np.asarray(task._TOKEN_EMBEDDINGS["TARGET"], dtype=np.float32)
    for b in batches:
        keys4 = np.concatenate([e_target[None, :], b.keys], axis=0)  # (4, d)
        ms, beta = _gate_value(b.query, keys4)
        g4 = gated_attention(b.query, keys4)
        demo["target_injected"].append({
            "alpha": b.alpha, "max_score": ms, "beta": beta,
            "gated": g4.tolist(), "weight_on_target": float(g4[0]),
            "labels": ["TARGET", "A", "B", "C"],
        })

    with open(run_dir / "demo.json", "w") as f:
        json.dump(demo, f, indent=2)

    c0 = demo["canonical"][0]
    print(f"[pass_2] canonical alpha=0: gated max_weight={c0['gated_mw']:.4f} "
          f"(regret={c0['gated_mw'] - 1/3:.4f}) | "
          f"softmax_raw={c0['softmax_raw_mw']:.3f} linear={c0['linear_mw']:.3f}")
    ti1 = demo["target_injected"][-1]
    print(f"[pass_2] target injected alpha=1: weight_on_target={ti1['weight_on_target']:.3f} "
          f"beta={ti1['beta']:.2f}")
    print(f"[pass_2] artefacts -> {run_dir}")


if __name__ == "__main__":
    main()
