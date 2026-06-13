"""Benchmark for the `attention_matrix_chain` goal.

Pure Python / no attempt-directory imports. Consumes the payload returned by
`task.evaluate` and emits a flat dict of scalar metrics.
"""

import math

VERSION = 1

# Pipeline-only hook: attempts run on the GPU (smoke test runs task/benchmark
# on CPU/NumPy).
GPU_REQUIREMENT = 1


def _alpha_key(prefix: str, alpha: float) -> str:
    """Slice key, e.g. ('chain_fidelity', 0.3) -> 'chain_fidelity_alpha_0p3'."""
    return f"{prefix}_alpha_{alpha:.1f}".replace(".", "p")


def score(payload: dict) -> dict[str, float | int]:
    """Compute flat scalar metrics from a `task.evaluate` payload.

    Expected payload keys:
        version (int == 1), num_heads (int), seq_len (int),
        canonical_alpha (float), alpha_sweep (list[float]),
        sweep (list[record]), single_hop_baseline (list[record]).

    Each sweep record:  {alpha, chain_fidelity, row_kl, n_seeds}.
    Each baseline record: {alpha, chain_fidelity, n_seeds}.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    required = ["version", "canonical_alpha", "alpha_sweep",
                "sweep", "single_hop_baseline"]
    for k in required:
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(
            f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(
            f"Unsupported payload version: {version}. Expected {VERSION}.")

    alpha_sweep = payload["alpha_sweep"]
    if not isinstance(alpha_sweep, (list, tuple)) or len(alpha_sweep) == 0:
        raise ValueError("payload['alpha_sweep'] must be a non-empty list")
    alpha_sweep = [float(a) for a in alpha_sweep]

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)) or len(sweep) != len(alpha_sweep):
        raise ValueError(
            "payload['sweep'] must be a list the same length as alpha_sweep")

    baseline = payload["single_hop_baseline"]
    if not isinstance(baseline, (list, tuple)) or len(baseline) != len(alpha_sweep):
        raise ValueError(
            "payload['single_hop_baseline'] must be a list the same length "
            "as alpha_sweep")

    def _index(records, what):
        out = {}
        for rec in records:
            if not isinstance(rec, dict):
                raise ValueError(f"Each {what} record must be a dict")
            if "alpha" not in rec:
                raise KeyError(f"{what} record missing 'alpha'")
            out[round(float(rec["alpha"]), 6)] = rec
        return out

    sweep_by_alpha = _index(sweep, "sweep")
    base_by_alpha = _index(baseline, "single_hop_baseline")

    metrics: dict[str, float | int] = {"version": VERSION}

    fidelity_vals = []
    for alpha in alpha_sweep:
        key = round(alpha, 6)
        srec = sweep_by_alpha.get(key, {})
        brec = base_by_alpha.get(key, {})

        fid = float(srec.get("chain_fidelity", 0.0))
        kl = float(srec.get("row_kl", 0.0))
        bfid = float(brec.get("chain_fidelity", 0.0))

        metrics[_alpha_key("chain_fidelity", alpha)] = fid
        metrics[_alpha_key("row_kl", alpha)] = kl
        metrics[_alpha_key("single_hop_baseline_fidelity", alpha)] = bfid
        metrics[_alpha_key("lift_over_baseline", alpha)] = fid - bfid

        fidelity_vals.append(fid)

    # --- Canonical condition ---
    canonical_alpha = float(payload["canonical_alpha"])
    canon_fid_key = _alpha_key("chain_fidelity", canonical_alpha)
    canon_base_key = _alpha_key("single_hop_baseline_fidelity", canonical_alpha)
    metrics["chain_fidelity_canonical"] = float(metrics.get(canon_fid_key, 0.0))
    canon_base = float(metrics.get(canon_base_key, 0.0))
    metrics["lift_over_baseline_canonical"] = (
        metrics["chain_fidelity_canonical"] - canon_base
    )

    # --- Mean fidelity across the whole sweep ---
    metrics["chain_fidelity_mean"] = (
        float(sum(fidelity_vals) / len(fidelity_vals)) if fidelity_vals else 0.0
    )

    # --- Headline: composition_robustness ---
    # Fraction of the easy-case (most uniform, last alpha) fidelity that
    # survives in the hard case (most peaked, first alpha), clipped to [0, 1].
    # A real composition mechanism stays high when rows are peaked; a single-hop
    # shortcut collapses there.
    fid_hard = float(metrics.get(_alpha_key("chain_fidelity", alpha_sweep[0]), 0.0))
    fid_easy = float(metrics.get(_alpha_key("chain_fidelity", alpha_sweep[-1]), 0.0))
    if fid_easy > 1e-12:
        robustness = fid_hard / fid_easy
    else:
        robustness = 0.0
    metrics["composition_robustness"] = float(max(0.0, min(1.0, robustness)))

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Pipeline hook: True if metrics are clearly degenerate (skip the jury).

    Never True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # A real composition mechanism must beat the single-hop baseline at the
    # canonical condition. Failing that is mechanically degenerate.
    fid = metrics.get("chain_fidelity_canonical")
    base = metrics.get("single_hop_baseline_fidelity_alpha_0p3")
    if isinstance(fid, (int, float)) and isinstance(base, (int, float)):
        if fid <= base:
            return True

    return False
