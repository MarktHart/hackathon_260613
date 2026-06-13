"""attention_lcs / pass_2 — hand-built LCS-alignment attention circuit.

Mechanism (a small delta from base_model.py's attention):

    score_h[i, j] = s_tok[h] * 1[A_i == B_j]  -  s_pos[h] * |i - j|

i.e. a single cross-attention head whose logits are the sum of
  (1) a *token-identity match* term — the dot product of one-hot token
      embeddings, which fires exactly when query symbol A_i equals key
      symbol B_j, and
  (2) an ALiBi/T5-style *relative-position bias* that prefers keys near the
      diagonal (the monotone-alignment prior of an LCS).

No training, no random weights: every coefficient is hand-set. The four heads
form a built-in ABLATION LADDER so the sweep itself is the causal evidence:

  head 0  token + position   (the full circuit)
  head 1  token only         (ablate the position bias)
  head 2  position only      (ablate the token gate)
  head 3  uniform            (ablate both -> the random baseline)

Everything runs in torch on CUDA.
"""

import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

# (s_tok, s_pos) per head. head 0 is the working circuit; 1/2/3 are ablations.
HEAD_PARAMS = [
    (30.0, 1.5),   # token + position
    (30.0, 0.0),   # token only
    (0.0, 1.5),    # position only
    (0.0, 0.0),    # uniform
]
HEAD_NAMES = ["token+pos (full)", "token only", "pos only", "uniform"]


# --------------------------------------------------------------------------
# Core hand-built attention, on the GPU.
# --------------------------------------------------------------------------
def _attention(seq_a_t: torch.Tensor, seq_b_t: torch.Tensor) -> torch.Tensor:
    """seq_a_t, seq_b_t: [B, L] long on DEVICE -> attn [B, n_heads, L, L]."""
    B, L = seq_a_t.shape
    match = (seq_a_t[:, :, None] == seq_b_t[:, None, :]).float()        # [B,L,L]
    pos = torch.arange(L, device=DEVICE)
    dist = (pos[:, None] - pos[None, :]).abs().float()[None]           # [1,L,L]

    heads = []
    for s_tok, s_pos in HEAD_PARAMS:
        logits = s_tok * match - s_pos * dist                          # [B,L,L]
        heads.append(torch.softmax(logits, dim=-1))
    return torch.stack(heads, dim=1)                                    # [B,H,L,L]


def model_fn(seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
    a = torch.as_tensor(seq_a, dtype=torch.long, device=DEVICE)
    b = torch.as_tensor(seq_b, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        attn = _attention(a, b)
    return attn.detach().cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------
# Local LCS + scoring (re-implements task.py's DP) for the operating-range
# sweep over vocab sizes the canonical generator does not expose.
# --------------------------------------------------------------------------
def _lcs_alignment(seq_a: np.ndarray, seq_b: np.ndarray) -> list[list[int]]:
    n, m = len(seq_a), len(seq_b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        ai = seq_a[i - 1]
        for j in range(1, m + 1):
            if ai == seq_b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            elif dp[i - 1, j] >= dp[i, j - 1]:
                dp[i, j] = dp[i - 1, j]
            else:
                dp[i, j] = dp[i, j - 1]
    match_keys = [[] for _ in range(n)]
    i, j = n, m
    while i > 0 and j > 0:
        if seq_a[i - 1] == seq_b[j - 1]:
            match_keys[i - 1].append(j - 1)
            i -= 1
            j -= 1
        elif dp[i - 1, j] >= dp[i, j - 1]:
            i -= 1
        else:
            j -= 1
    for ml in match_keys:
        ml.sort()
    return match_keys


def _score_mass(attn_h: np.ndarray, match_keys: list[list[list[int]]], k_len: int):
    """Mean attention mass on LCS keys + the uniform baseline mass."""
    mass_total = 0.0
    base_total = 0.0
    n = 0
    for b, mk in enumerate(match_keys):
        for q, keys in enumerate(mk):
            if not keys:
                continue
            ka = np.asarray(keys, dtype=np.int64)
            mass_total += float(attn_h[b, q, ka].sum())
            base_total += ka.shape[0] / k_len
            n += 1
    if n == 0:
        return 0.0, 0.0, 0
    return mass_total / n, base_total / n, n


def operating_range(seq_len=16, num=128):
    """Head-0 lift vs vocabulary size — where the circuit works and breaks."""
    vocabs = [2, 4, 8, 16, 32, 64]
    out = {"vocab_sizes": vocabs, "mass": [], "lift": [], "baseline": [], "seq_len": seq_len}
    for v in vocabs:
        rng = np.random.default_rng(1234 + v)
        a = rng.integers(0, v, size=(num, seq_len), dtype=np.int32)
        b = rng.integers(0, v, size=(num, seq_len), dtype=np.int32)
        mk = [_lcs_alignment(a[i], b[i]) for i in range(num)]
        attn = model_fn(a, b)              # [num, H, L, L] on GPU then back
        mass, base, _ = _score_mass(attn[:, 0], mk, seq_len)
        out["mass"].append(mass)
        out["baseline"].append(base)
        out["lift"].append(mass - base)
    return out


def main():
    task = load_task(__file__)

    # --- canonical benchmark (the only condition that lands in benchmark.json) ---
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    # --- operating-range sweep (custom artefact) ---
    orr = operating_range()
    (run_dir / "operating_range.json").write_text(json.dumps(orr, indent=2))

    # --- sample heatmap artefact: richest-LCS example from the canonical batch ---
    batch = task.generate(seed=0)
    counts = [sum(1 for k in mk if k) for mk in batch.match_keys]
    idx = int(np.argmax(counts))
    a = batch.seq_a[idx:idx + 1]
    b = batch.seq_b[idx:idx + 1]
    attn = model_fn(a, b)                  # [1, H, L, L]
    sample = {
        "index": idx,
        "seq_a": batch.seq_a[idx].tolist(),
        "seq_b": batch.seq_b[idx].tolist(),
        "match_keys": batch.match_keys[idx],
        "attn_full": attn[0, 0].tolist(),     # token+pos head
        "attn_tokonly": attn[0, 1].tolist(),  # token-only ablation
        "head_names": HEAD_NAMES,
    }
    (run_dir / "sample.json").write_text(json.dumps(sample))

    print("random_baseline_mass:", round(payload["random_baseline_mass"], 4))
    for rec, name in zip(payload["sweep"], HEAD_NAMES):
        print(f"  head {rec['head']} [{name}]: mass={rec['lcs_attention_mass']:.4f} "
              f"lift={rec['lcs_lift']:.4f}")
    print("operating range lift:", [round(x, 3) for x in orr["lift"]])


if __name__ == "__main__":
    main()
