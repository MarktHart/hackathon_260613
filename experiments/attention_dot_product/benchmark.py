"""Benchmark for the `attention_dot_product` goal.

Pure Python, deterministic, side-effect free.  Consumes the payload produced by
`task.evaluate` and returns a flat dict of scalar metrics.

Headline: `attention_fidelity` — fraction of the uniform-attention baseline
error that the attempt removes, averaged over the sequence-length sweep, in
[0, 1].  Bigger is better.
"""

import math

VERSION = 1
GPU_REQUIREMENT = 1  # attempts run on the GPU; task/benchmark stay CPU/NumPy


def _seqlen_suffix(seq_len: int) -> str:
    return f"seqlen_{int(seq_len)}"


def _mean(xs) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else 0.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _validate(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")
    if payload.get("version") != VERSION:
        raise ValueError(
            f"Unsupported payload version: {payload.get('version')!r}; "
            f"expected {VERSION}"
        )
    for key in ("config", "sweep"):
        if key not in payload:
            raise KeyError(f"Missing payload key: {key!r}")

    cfg = payload["config"]
    if not isinstance(cfg, dict):
        raise ValueError("payload['config'] must be a dict")
    for k in ("d_head", "n_heads", "canonical_seq_len", "seq_len_sweep"):
        if k not in cfg:
            raise KeyError(f"Missing config key: {k!r}")

    sweep = payload["sweep"]
    if not isinstance(sweep, list) or not sweep:
        raise ValueError("payload['sweep'] must be a non-empty list")
    for i, rec in enumerate(sweep):
        if not isinstance(rec, dict):
            raise ValueError(f"sweep[{i}] must be a dict")
        for k in ("seq_len", "mse", "rel_error", "cos_sim", "baseline_mse"):
            if k not in rec:
                raise KeyError(f"sweep[{i}] missing key: {k!r}")


def score(payload: dict) -> dict[str, float | int]:
    """Compute metrics from a `task.evaluate` payload."""
    _validate(payload)

    cfg = payload["config"]
    sweep = payload["sweep"]
    canonical_seq_len = int(cfg["canonical_seq_len"])

    metrics: dict[str, float | int] = {"version": VERSION}

    by_seqlen: dict[int, dict] = {}
    for rec in sweep:
        L = int(rec["seq_len"])
        mse = float(rec["mse"])
        rel = float(rec["rel_error"])
        cos = float(rec["cos_sim"])
        base = float(rec["baseline_mse"])
        by_seqlen[L] = {"mse": mse, "rel_error": rel, "cos_sim": cos, "baseline_mse": base}

        suf = _seqlen_suffix(L)
        metrics[f"mse_{suf}"] = mse
        metrics[f"rel_error_{suf}"] = rel
        metrics[f"cos_sim_{suf}"] = cos
        metrics[f"baseline_mse_{suf}"] = base

    # --- canonical (default condition) --------------------------------------
    canon = by_seqlen.get(canonical_seq_len)
    if canon is None:
        raise ValueError(
            f"No sweep record at canonical_seq_len={canonical_seq_len}"
        )
    metrics["mse_canonical"] = canon["mse"]
    metrics["rel_error_canonical"] = canon["rel_error"]
    metrics["cos_sim_canonical"] = canon["cos_sim"]
    metrics["baseline_mse_canonical"] = canon["baseline_mse"]
    if canon["baseline_mse"] > 1e-12:
        metrics["attention_fidelity_canonical"] = _clamp01(
            1.0 - canon["mse"] / canon["baseline_mse"]
        )
    else:
        metrics["attention_fidelity_canonical"] = 0.0
    metrics["lift_over_baseline_canonical"] = canon["baseline_mse"] - canon["mse"]

    # --- headline: variance-explained fidelity over the whole sweep ----------
    mean_mse = _mean(v["mse"] for v in by_seqlen.values())
    mean_base = _mean(v["baseline_mse"] for v in by_seqlen.values())
    if mean_base > 1e-12:
        metrics["attention_fidelity"] = _clamp01(1.0 - mean_mse / mean_base)
    else:
        metrics["attention_fidelity"] = 0.0

    # --- robustness: worst-case cosine across the sweep ----------------------
    cos_vals = [v["cos_sim"] for v in by_seqlen.values()]
    worst = min(cos_vals) if cos_vals else 0.0
    metrics["cos_sim_worst"] = float(worst)
    metrics["cos_sim_mean"] = _mean(cos_vals)
    metrics["attention_robustness"] = _clamp01(worst)

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Mechanical degeneracy check; short-circuits the jury when True."""
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # No better than the uniform-attention baseline anywhere in the sweep.
    fidelity = metrics.get("attention_fidelity")
    if isinstance(fidelity, (int, float)) and fidelity <= 0.0:
        return True

    # Worst-case output essentially uncorrelated with the true attention output.
    worst = metrics.get("cos_sim_worst")
    if isinstance(worst, (int, float)) and worst < 0.1:
        return True

    return False
