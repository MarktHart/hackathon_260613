"""attention_one_hot — pass_5

Hand-built scaled dot-product attention head (the canonical one-hot lookup
primitive) PLUS a measured strawman / ablation suite that the previous attempt
was missing.

What is new vs pass_4:
  1. A *measured* baseline comparison: the no-temperature (tau=1) softmax,
     the no-attention uniform head, and a causally query-patched head are all
     run through the SAME task.evaluate evaluator, so we can show "the method
     works while the strawmen fail" with real numbers instead of an analytic
     1/L line.
  2. A causal / faithfulness check: corrupt (patch) the query vector — the
     activation that carries the lookup key — and watch one-hot collapse to
     uniform. This is the activation-patching evidence that the query.key dot
     product is the load-bearing wire.
  3. A query-key alignment robustness sweep (under realistic, NON-orthogonal
     noise keys) that reveals the operating range and shows why the exp
     non-linearity and the sharp temperature each matter.

Everything computes in torch on CUDA, as required.
"""

import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

# A fixed, deterministic "wrong" query used for the causal patching ablation.
_BAD_Q = np.random.default_rng(777).normal(size=32).astype(np.float32)
_BAD_Q = _BAD_Q / np.linalg.norm(_BAD_Q)


# ---------------------------------------------------------------------------
# Core GPU attention primitive + variants
# ---------------------------------------------------------------------------
def _attn(query: np.ndarray, keys: np.ndarray, temperature: float, mode: str) -> np.ndarray:
    """Compute an attention distribution on the GPU for the given variant."""
    qt = torch.as_tensor(query, dtype=torch.float32, device=DEVICE)
    kt = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)
    L = kt.shape[0]

    scores = kt @ qt  # (L,) raw dot-product similarity

    if mode == "method":
        attn = torch.softmax(scores / temperature, dim=-1)
    elif mode == "no_temperature":
        # Ablate the sharp temperature: plain softmax(QK^T).
        attn = torch.softmax(scores / 1.0, dim=-1)
    elif mode == "linear_no_exp":
        # Ablate the exp non-linearity: normalise rectified raw scores.
        w = torch.relu(scores / temperature)
        s = w.sum()
        attn = w / s if float(s) > 0 else torch.full((L,), 1.0 / L, device=DEVICE)
    elif mode == "uniform":
        # Ablate attention entirely: a no-mechanism uniform head.
        attn = torch.full((L,), 1.0 / L, device=DEVICE)
    else:
        raise ValueError(f"unknown mode {mode}")

    return attn.detach().cpu().numpy()


# The official method handed to the benchmark.
def method_fn(query: np.ndarray, keys: np.ndarray, temperature: float) -> np.ndarray:
    return _attn(query, keys, temperature, "method")


def no_temperature_fn(query, keys, temperature):
    return _attn(query, keys, temperature, "no_temperature")


def uniform_fn(query, keys, temperature):
    return _attn(query, keys, temperature, "uniform")


def corrupted_query_fn(query, keys, temperature):
    """Causal patch: replace the query activation with a fixed wrong vector."""
    return _attn(_BAD_Q, keys, temperature, "method")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sweep_target_attention(model_fn) -> dict:
    """Run a model_fn through the task evaluator, return {length: target_attention}."""
    payload = task.evaluate(model_fn)
    return {rec["length"]: rec["target_attention"] for rec in payload["sweep"]}


def alignment_sweep(d_model=32, L=64, temperature=0.1, n_alpha=11, seed=123):
    """Query-key alignment robustness sweep under REALISTIC (non-orthogonal) noise.

    Unlike the task's idealised orthogonal noise keys, here every noise key is a
    fresh random unit vector (small but non-zero overlap with the query). The
    target key is target = alpha*q + sqrt(1-alpha^2)*orth, so alpha is the
    cosine match between query and the needle. This stress test is where the
    exp / temperature components actually start to matter, exposing each
    variant's operating range and breaking point.
    """
    rng = np.random.default_rng(seed)
    q = rng.normal(size=d_model).astype(np.float32)
    q = q / np.linalg.norm(q)

    # A unit vector orthogonal to q, used to tilt the target away from a match.
    o = rng.normal(size=d_model).astype(np.float32)
    o = o - (o @ q) * q
    o = o / np.linalg.norm(o)

    # Fixed realistic noise keys (full random units; NOT projected orthogonal).
    noise = rng.normal(size=(L, d_model)).astype(np.float32)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)

    target_pos = 0
    alphas = np.linspace(1.0, 0.0, n_alpha)
    modes = ["method", "no_temperature", "linear_no_exp", "uniform"]
    out = {"alphas": [float(a) for a in alphas]}
    for m in modes:
        out[m] = []

    for a in alphas:
        keys = noise.copy()
        tgt = a * q + np.sqrt(max(0.0, 1.0 - a * a)) * o
        tgt = tgt / np.linalg.norm(tgt)
        keys[target_pos] = tgt.astype(np.float32)
        for m in modes:
            attn = _attn(q, keys, temperature, m)
            out[m].append(float(attn[target_pos]))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
task = load_task(__file__)


def main():
    run_dir = results_dir(__file__)

    # 1) Official benchmark: the hand-built dot-product head.
    payload = task.evaluate(method_fn)
    record_benchmark(__file__, run_dir, payload)

    # 2) Measured strawman / ablation suite over the canonical length sweep,
    #    run through the SAME evaluator the headline uses.
    variants = {
        "method": method_fn,
        "no_temperature": no_temperature_fn,
        "corrupted_query": corrupted_query_fn,
        "uniform": uniform_fn,
    }
    variant_sweeps = {name: sweep_target_attention(fn) for name, fn in variants.items()}

    # 3) Alignment robustness / operating-range sweep under realistic noise.
    align = alignment_sweep()

    ablations = {
        "canonical_length": payload["canonical_length"],
        "temperature": payload["temperature"],
        "d_model": payload["d_model"],
        "lengths": [rec["length"] for rec in payload["sweep"]],
        "variant_target_attention": variant_sweeps,
        "uniform_baseline": {rec["length"]: 1.0 / rec["length"] for rec in payload["sweep"]},
        "alignment_sweep": align,
        "notes": {
            "method": "softmax(QK^T / tau), tau=0.1 — the hand-built one-hot head",
            "no_temperature": "softmax(QK^T) — exp kept, sharp temperature ablated",
            "corrupted_query": "method but query activation patched to a fixed wrong vector (causal check)",
            "uniform": "no attention mechanism — uniform 1/L head",
            "linear_no_exp": "normalise relu(QK^T/tau) — exp non-linearity ablated (alignment sweep only)",
        },
    }
    (run_dir / "ablations.json").write_text(json.dumps(ablations, indent=2))

    # Console summary.
    print("== canonical L=64 target attention ==")
    for name, sw in variant_sweeps.items():
        print(f"  {name:16s}: {sw[64]:.4f}")
    print(f"  uniform 1/L      : {1.0/64:.4f}")


if __name__ == "__main__":
    main()
