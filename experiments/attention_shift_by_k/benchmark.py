"""Benchmark for the attention_shift_by_k goal.

Consumes the payload from ``task.evaluate`` and returns a flat dict of scalar
metrics. Pure, deterministic, side-effect free. No imports from any attempt
directory.

Payload contract (see README.md):

    {
      "version": 1,
      "model_name": str,
      "seq_len": int,
      "batch_size": int,
      "vocab_size": int,
      "num_heads": int,
      "k_values": list[int],
      "canonical_k": int,
      "uniform_baseline": float,         # 1 / seq_len
      "sweep": [
        {
          "k": int,
          "best_head_index": int,
          "best_head_mass": float,       # mean attn on key i-k, best head
          "best_head_argmax_acc": float, # fraction of queries peaking at i-k
          "mean_head_mass": float,       # across-head average mass
          "uniform_baseline": float,     # 1 / seq_len
        },
        ...
      ],
    }
"""

from __future__ import annotations

import math
from typing import Any

VERSION = 1


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _finite(x: Any) -> bool:
    return _is_num(x) and not math.isnan(x) and not math.isinf(x)


def score(payload: dict) -> dict[str, float | int]:
    # ---- Input validation ----
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    if payload.get("version") != VERSION:
        raise ValueError(
            f"payload version {payload.get('version')!r} != benchmark VERSION {VERSION}"
        )

    sweep = payload.get("sweep")
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("payload['sweep'] must be a non-empty list")

    canonical_k = payload.get("canonical_k")
    if not isinstance(canonical_k, int) or isinstance(canonical_k, bool):
        raise ValueError(f"payload['canonical_k'] must be an int, got {canonical_k!r}")

    uniform_baseline = payload.get("uniform_baseline")
    if not _finite(uniform_baseline) or not (0.0 < uniform_baseline < 1.0):
        raise ValueError(
            f"payload['uniform_baseline'] must be in (0, 1), got {uniform_baseline!r}"
        )

    metrics: dict[str, float | int] = {"version": VERSION}

    norm_lifts: list[float] = []
    canonical_mass: float | None = None
    seen_k: set[int] = set()

    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError(f"sweep records must be dicts, got {rec!r}")
        k = rec.get("k")
        if not isinstance(k, int) or isinstance(k, bool):
            raise ValueError(f"sweep record 'k' must be an int, got {k!r}")
        if k in seen_k:
            raise ValueError(f"duplicate k={k} in sweep")
        seen_k.add(k)

        mass = rec.get("best_head_mass")
        argmax = rec.get("best_head_argmax_acc")
        mean_mass = rec.get("mean_head_mass")
        base = rec.get("uniform_baseline", uniform_baseline)
        for name, val in (
            ("best_head_mass", mass),
            ("best_head_argmax_acc", argmax),
            ("mean_head_mass", mean_mass),
            ("uniform_baseline", base),
        ):
            if not _finite(val):
                raise ValueError(f"sweep k={k}: '{name}' must be finite, got {val!r}")

        tag = str(int(k))
        metrics[f"shift_mass_k_{tag}"] = float(mass)
        metrics[f"shift_argmax_acc_k_{tag}"] = float(argmax)
        metrics[f"mean_head_mass_k_{tag}"] = float(mean_mass)
        metrics[f"linear_baseline_mass_k_{tag}"] = float(base)

        # Chance-normalised lift in [0, 1]: 0 = at baseline, 1 = all mass on target.
        denom = 1.0 - float(base)
        norm = (float(mass) - float(base)) / denom if denom > 0 else 0.0
        norm = max(0.0, min(1.0, norm))
        metrics[f"shift_lift_k_{tag}"] = norm
        norm_lifts.append(norm)

        if k == canonical_k:
            canonical_mass = float(mass)
            metrics["shift_mass_canonical"] = float(mass)
            metrics["shift_argmax_acc_canonical"] = float(argmax)
            metrics["lift_over_baseline_canonical"] = float(mass) - float(base)

    if canonical_mass is None:
        raise ValueError(
            f"canonical_k={canonical_k} not present among swept k values {sorted(seen_k)}"
        )

    # ---- Headline summary (bigger is better, in [0, 1]) ----
    # How robustly a single head implements shift-by-k across all offsets.
    metrics["shift_robustness"] = float(sum(norm_lifts) / len(norm_lifts))
    metrics["uniform_baseline"] = float(uniform_baseline)

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Mechanical degeneracy check; skips the (expensive) jury when True.

    True only when the result is mechanically degenerate: NaN/inf math, or the
    canonical-condition best head fails to beat the uniform baseline by >10%.
    Never True for a borderline-but-real shift-by-k head.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    mass = metrics.get("shift_mass_canonical")
    baseline = metrics.get("uniform_baseline")
    if not _is_num(mass) or not _is_num(baseline):
        return True
    if mass <= baseline * 1.1:
        return True

    return False


GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU
