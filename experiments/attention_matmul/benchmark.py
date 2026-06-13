"""Benchmark for the `attention_matmul` goal.

Pure Python. No imports from any attempt directory. Deterministic and
side-effect free. See README.md for the metric definitions.
"""

import math

VERSION = 1

# Pipeline-only hook: every attempt runs on the GPU.
GPU_REQUIREMENT = 1

CONDITIONS = ["orthogonal", "cos_0p3", "cos_0p7", "uniform"]
CANONICAL_CONDITION = "cos_0p3"

_EPS = 1e-12
_KL_CLIP = 1e6


def _fidelity(value_cond: float, value_base: float) -> float:
    """Fraction of baseline error removed, clipped to [0, 1].

    0.0 == no better than the uniform baseline; 1.0 == perfect (zero error).
    Edge case: a (near-)zero baseline error means the baseline is already
    perfect, so normalisation is ill-defined — return 1.0.
    """
    if not math.isfinite(value_cond) or not math.isfinite(value_base):
        return 0.0
    if value_base <= _EPS:
        return 1.0
    f = 1.0 - (value_cond / value_base)
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


def score(payload: dict) -> dict[str, float | int]:
    """Compute flat scalar metrics from the payload returned by task.evaluate().

    Expected payload keys:
        version (int == 1), config (dict), canonical_condition (str),
        conditions (list[str]), sweep (list[record]),
        linear_baseline (dict[str, record]).

    Each sweep record: {qk_alignment, output_mse, attribution_kl, rowsum_mae}.
    Each linear_baseline record: {output_mse, attribution_kl, rowsum_mae}.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    for k in ("version", "sweep", "linear_baseline"):
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(
            f"payload['version'] must be int, got {type(version).__name__}"
        )
    if version != VERSION:
        raise ValueError(
            f"Unsupported payload version: {version}. Expected {VERSION}."
        )

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)) or len(sweep) == 0:
        raise ValueError("payload['sweep'] must be a non-empty list")

    baseline = payload["linear_baseline"]
    if not isinstance(baseline, dict) or len(baseline) == 0:
        raise ValueError("payload['linear_baseline'] must be a non-empty dict")

    # Index sweep records by condition.
    sweep_by_cond: dict[str, dict] = {}
    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError("each sweep record must be a dict")
        if "qk_alignment" not in rec:
            raise KeyError("sweep record missing 'qk_alignment'")
        sweep_by_cond[str(rec["qk_alignment"])] = rec

    metrics: dict[str, float | int] = {"version": VERSION}

    fidelity_vals: list[float] = []

    for cond in CONDITIONS:
        srec = sweep_by_cond.get(cond)
        brec = baseline.get(cond)
        if srec is None or brec is None:
            # Condition absent from this payload: record neutral values so the
            # dashboard still has a slot, but don't fold into the headline.
            metrics[f"attribution_fidelity_qk_{cond}"] = 0.0
            metrics[f"output_reconstruction_qk_{cond}"] = 0.0
            metrics[f"rowsum_mae_qk_{cond}"] = 0.0
            metrics[f"linear_baseline_attribution_fidelity_qk_{cond}"] = 0.0
            metrics[f"linear_baseline_output_reconstruction_qk_{cond}"] = 0.0
            continue

        kl_cond = float(srec.get("attribution_kl", _KL_CLIP))
        kl_base = float(brec.get("attribution_kl", 0.0))
        mse_cond = float(srec.get("output_mse", float("inf")))
        mse_base = float(brec.get("output_mse", 0.0))
        rowsum = float(srec.get("rowsum_mae", 0.0))

        if not math.isfinite(kl_cond):
            kl_cond = _KL_CLIP
        elif kl_cond > _KL_CLIP:
            kl_cond = _KL_CLIP

        attr_fid = _fidelity(kl_cond, kl_base)
        out_rec = _fidelity(mse_cond, mse_base)

        metrics[f"attribution_fidelity_qk_{cond}"] = attr_fid
        metrics[f"output_reconstruction_qk_{cond}"] = out_rec
        metrics[f"rowsum_mae_qk_{cond}"] = rowsum if math.isfinite(rowsum) else _KL_CLIP
        # Baseline normalised against itself is 0.0 by construction.
        metrics[f"linear_baseline_attribution_fidelity_qk_{cond}"] = 0.0
        metrics[f"linear_baseline_output_reconstruction_qk_{cond}"] = 0.0

        fidelity_vals.append(attr_fid)

    # Headline: attribution fidelity at the canonical condition.
    canonical = str(payload.get("canonical_condition", CANONICAL_CONDITION))
    metrics["attribution_fidelity_canonical"] = float(
        metrics.get(f"attribution_fidelity_qk_{canonical}", 0.0)
    )
    metrics["output_reconstruction_canonical"] = float(
        metrics.get(f"output_reconstruction_qk_{canonical}", 0.0)
    )

    # Mean fidelity across all present conditions — a coarse aggregate.
    if fidelity_vals:
        metrics["attribution_fidelity_mean"] = float(
            sum(fidelity_vals) / len(fidelity_vals)
        )
    else:
        metrics["attribution_fidelity_mean"] = 0.0

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """True when metrics are mechanically degenerate, to skip the jury.

    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # Fidelity is clipped to [0, 1] with 0.0 == no better than the uniform
    # baseline. An attempt that fails to beat the baseline at the canonical
    # condition is mechanically degenerate.
    canonical = metrics.get("attribution_fidelity_canonical")
    if isinstance(canonical, (int, float)) and canonical <= 0.0:
        return True

    return False
