"""
Data generation and evaluation for the `attention_viterbi` goal.

Exports:
    generate(seed=0) -> Batch
    evaluate(model_fn) -> payload_dict      (shape consumed by benchmark.score)
    random_model_fn() -> callable matching the model_fn signature

Pure Python / NumPy. No I/O, no network, no torch, no scipy.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

# ---------------------------------------------------------------------------
# Canonical HMM parameters (frozen). Changing any of these is a VERSION bump.
# ---------------------------------------------------------------------------
HMM_PI = np.array([0.6, 0.3, 0.1], dtype=np.float64)
HMM_A = np.array([
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.2, 0.3, 0.5],
], dtype=np.float64)
HMM_B = np.array([
    [0.8, 0.10, 0.05, 0.05],
    [0.1, 0.70, 0.10, 0.10],
    [0.05, 0.15, 0.70, 0.10],
], dtype=np.float64)

N_STATES = 3
N_OBS = 4
SEQ_LEN = 20          # includes the BOS/first token at position 0
N_EVAL_SEQ = 100
EVAL_SEED = 42        # canonical evaluation set

# Model config (canonical). Attempts must produce attention of this shape.
MODEL_CONFIG = {
    "n_layers": 2,
    "n_heads": 4,
    "d_model": 64,
    "seq_len": SEQ_LEN,
    "vocab_size": N_OBS,
}

HMM_CONFIG = {
    "n_states": N_STATES,
    "n_obs": N_OBS,
    "pi": HMM_PI.tolist(),
    "A": HMM_A.tolist(),
    "B": HMM_B.tolist(),
}

# The model_fn the attempt must expose:
#   model_fn(input_ids: np.ndarray[batch, seq_len]) -> {
#       "attn_weights": np.ndarray[batch, n_layers, n_heads, seq_len, seq_len],
#       "logits":       np.ndarray[batch, seq_len, vocab_size],
#   }
ModelFn = Callable[[np.ndarray], Dict[str, np.ndarray]]


@dataclass(frozen=True)
class Batch:
    """The fixed evaluation set."""
    input_ids: np.ndarray          # [N_EVAL_SEQ, SEQ_LEN], int32 (observations 0..3)
    viterbi_paths: np.ndarray      # [N_EVAL_SEQ, SEQ_LEN], int32 (states 0..2)


# ---------------------------------------------------------------------------
# Viterbi decoder
# ---------------------------------------------------------------------------
def _viterbi(obs_seq: np.ndarray) -> np.ndarray:
    """Most likely state sequence for one observation sequence (log-space)."""
    T = len(obs_seq)
    log_pi = np.log(HMM_PI + 1e-12)
    log_A = np.log(HMM_A + 1e-12)
    log_B = np.log(HMM_B + 1e-12)

    delta = np.full((T, N_STATES), -np.inf, dtype=np.float64)
    psi = np.zeros((T, N_STATES), dtype=np.int32)

    delta[0] = log_pi + log_B[:, obs_seq[0]]
    for t in range(1, T):
        for s in range(N_STATES):
            scores = delta[t - 1] + log_A[:, s]
            psi[t, s] = int(np.argmax(scores))
            delta[t, s] = float(np.max(scores)) + log_B[s, obs_seq[t]]

    state_seq = np.zeros(T, dtype=np.int32)
    state_seq[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        state_seq[t] = psi[t + 1, state_seq[t + 1]]
    return state_seq


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(seed: int = 0) -> Batch:
    """
    Deterministic evaluation batch. seed=EVAL_SEED (42) is the canonical
    condition every attempt is scored on; other seeds give ablation batches.
    """
    rng = np.random.default_rng(seed)
    input_ids = np.zeros((N_EVAL_SEQ, SEQ_LEN), dtype=np.int32)
    viterbi_paths = np.zeros((N_EVAL_SEQ, SEQ_LEN), dtype=np.int32)

    for i in range(N_EVAL_SEQ):
        states = np.zeros(SEQ_LEN, dtype=np.int32)
        states[0] = int(rng.choice(N_STATES, p=HMM_PI))
        for t in range(1, SEQ_LEN):
            states[t] = int(rng.choice(N_STATES, p=HMM_A[states[t - 1]]))

        obs = np.zeros(SEQ_LEN, dtype=np.int32)
        for t in range(SEQ_LEN):
            obs[t] = int(rng.choice(N_OBS, p=HMM_B[states[t]]))

        input_ids[i] = obs
        viterbi_paths[i] = _viterbi(obs)

    return Batch(input_ids=input_ids, viterbi_paths=viterbi_paths)


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------
def _excess_on_predecessor(head_attn: np.ndarray) -> float:
    """
    Mean *excess* attention placed on the Viterbi predecessor (position t-1).

    For a first-order HMM the Viterbi backpointer of query t is always t-1, so
    the relevant attention quantity is alpha[t, t-1] minus the uniform-over-past
    level mean(alpha[t, :t]). Bounded in (-1, 1); exactly 0 for uniform causal
    attention. Bigger = the head concentrates more mass on the predecessor than
    a uniform reader would.

    Args:
        head_attn: [batch, seq_len, seq_len] causal attention for one head.
    Returns:
        mean excess across all (batch, t=1..T-1) pairs.
    """
    batch, T, _ = head_attn.shape
    vals: List[float] = []
    for b in range(batch):
        for t in range(1, T):
            past = head_attn[b, t, :t]
            if past.size == 0:
                continue
            vals.append(float(head_attn[b, t, t - 1]) - float(np.mean(past)))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _excess_by_position(head_attn: np.ndarray) -> List[Dict[str, Any]]:
    """Per-query-position mean excess attention on the predecessor."""
    batch, T, _ = head_attn.shape
    out: List[Dict[str, Any]] = []
    for t in range(1, T):
        vals: List[float] = []
        for b in range(batch):
            past = head_attn[b, t, :t]
            if past.size == 0:
                continue
            vals.append(float(head_attn[b, t, t - 1]) - float(np.mean(past)))
        out.append({
            "pos": int(t),
            "excess": float(np.mean(vals)) if vals else 0.0,
            "n": len(vals),
        })
    return out


# ---------------------------------------------------------------------------
# Baseline attention generators
# ---------------------------------------------------------------------------
def _uniform_causal_attention(batch: int, seq_len: int) -> np.ndarray:
    attn = np.zeros((batch, seq_len, seq_len), dtype=np.float64)
    for t in range(1, seq_len):
        attn[:, t, :t] = 1.0 / t
    return attn


def _random_causal_attention(batch: int, seq_len: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    attn = np.zeros((batch, seq_len, seq_len), dtype=np.float64)
    for t in range(1, seq_len):
        attn[:, t, :t] = rng.dirichlet(np.ones(t), size=batch)
    return attn


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model_fn: ModelFn) -> Dict[str, Any]:
    """
    Run model_fn over the canonical batch and return the payload dict consumed
    by benchmark.score(). Attempts never build the payload themselves.
    """
    batch = generate(EVAL_SEED)
    input_ids = batch.input_ids

    out = model_fn(input_ids)
    if not isinstance(out, dict):
        raise ValueError(f"model_fn must return a dict, got {type(out).__name__}")
    if "attn_weights" not in out:
        raise KeyError("model_fn output missing 'attn_weights'")

    attn_weights = np.asarray(out["attn_weights"], dtype=np.float64)

    n_layers = MODEL_CONFIG["n_layers"]
    n_heads = MODEL_CONFIG["n_heads"]
    expected = (N_EVAL_SEQ, n_layers, n_heads, SEQ_LEN, SEQ_LEN)
    if attn_weights.shape != expected:
        raise ValueError(
            f"attn_weights shape {attn_weights.shape} != expected {expected}"
        )

    # Per-head excess attention on the Viterbi predecessor.
    per_head: List[Dict[str, Any]] = []
    best = {"excess": -np.inf, "layer": 0, "head": 0}
    for layer in range(n_layers):
        for head in range(n_heads):
            head_attn = attn_weights[:, layer, head, :, :]
            ex = _excess_on_predecessor(head_attn)
            per_head.append({"layer": layer, "head": head, "excess": ex})
            if ex > best["excess"]:
                best = {"excess": ex, "layer": layer, "head": head}

    # Per-position breakdown for the single strongest head.
    best_head_attn = attn_weights[:, best["layer"], best["head"], :, :]
    positional = _excess_by_position(best_head_attn)

    # Baselines under identical conditions.
    uniform_excess = _excess_on_predecessor(
        _uniform_causal_attention(N_EVAL_SEQ, SEQ_LEN)
    )
    random_excess = _excess_on_predecessor(
        _random_causal_attention(N_EVAL_SEQ, SEQ_LEN, seed=123)
    )

    return {
        "version": 1,
        "model_config": dict(MODEL_CONFIG),
        "hmm_config": dict(HMM_CONFIG),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "seq_len": SEQ_LEN,
        "eval_sequences": input_ids.tolist(),
        "viterbi_paths": batch.viterbi_paths.tolist(),
        "best_head": {"layer": int(best["layer"]), "head": int(best["head"])},
        "per_head": per_head,                       # 8 records
        "positional": positional,                   # SEQ_LEN-1 records
        "baseline_uniform_excess": float(uniform_excess),
        "baseline_random_excess": float(random_excess),
    }


# ---------------------------------------------------------------------------
# Random model (smoke test)
# ---------------------------------------------------------------------------
def random_model_fn() -> ModelFn:
    """
    A callable with the real model_fn signature whose body returns
    correctly-shaped, validly-normalised random/zero values. Pure NumPy.
    """
    n_layers = MODEL_CONFIG["n_layers"]
    n_heads = MODEL_CONFIG["n_heads"]
    seq_len = MODEL_CONFIG["seq_len"]
    vocab_size = MODEL_CONFIG["vocab_size"]
    rng = np.random.default_rng(0)

    def _fn(input_ids: np.ndarray) -> Dict[str, np.ndarray]:
        batch = int(np.asarray(input_ids).shape[0])
        # Valid causal, row-stochastic attention (uniform over the past).
        attn = np.zeros((batch, n_layers, n_heads, seq_len, seq_len), dtype=np.float32)
        for t in range(1, seq_len):
            attn[:, :, :, t, :t] = 1.0 / t
        logits = rng.standard_normal((batch, seq_len, vocab_size)).astype(np.float32)
        return {"attn_weights": attn, "logits": logits}

    return _fn
