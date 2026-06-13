import math

VERSION = 2

# Pipeline-only hook: how many GPU slots the attempt subprocess needs.
GPU_REQUIREMENT = 1


def _cos_key(prefix: str, cos_AB: float) -> str:
    """Slice key name, e.g. ('and_sharpness', 0.7) -> 'and_sharpness_cos_0p7'."""
    return f"{prefix}_cos_{cos_AB:.1f}".replace(".", "p")


def score(payload: dict) -> dict[str, float | int]:
    """
    Compute flat scalar metrics from the payload returned by task.evaluate().

    Expected payload keys:
        version (int == 2), d (int), canonical_cosine (float),
        cos_AB_sweep (list[float]), sweep (list[record]),
        linear_baseline (list[record]).

    Each sweep record: {cosine, and_sharpness, false_positive_rate,
                        false_negative_rate, n_seeds}.
    Each linear_baseline record: {cosine, and_sharpness, n_seeds}.
    """
    # --- Input validation ---
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    required_keys = ["version", "d", "canonical_cosine", "cos_AB_sweep",
                     "sweep", "linear_baseline"]
    for k in required_keys:
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    d = payload["d"]
    if not isinstance(d, (int, float)) or d <= 0:
        raise ValueError(f"payload['d'] must be a positive number, got {d!r}")

    cos_sweep = payload["cos_AB_sweep"]
    if not isinstance(cos_sweep, (list, tuple)) or len(cos_sweep) == 0:
        raise ValueError("payload['cos_AB_sweep'] must be a non-empty list")
    cos_sweep = [float(c) for c in cos_sweep]

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)) or len(sweep) != len(cos_sweep):
        raise ValueError("payload['sweep'] must be a list of same length as cos_AB_sweep")

    linear_baseline = payload["linear_baseline"]
    if not isinstance(linear_baseline, (list, tuple)) or len(linear_baseline) != len(cos_sweep):
        raise ValueError(
            "payload['linear_baseline'] must be a list of same length as cos_AB_sweep"
        )

    # --- Index records by nominal cosine ---
    def _index(records, what):
        out = {}
        for rec in records:
            if not isinstance(rec, dict):
                raise ValueError(f"Each {what} record must be a dict")
            if "cosine" not in rec:
                raise KeyError(f"{what} record missing 'cosine'")
            out[round(float(rec["cosine"]), 6)] = rec
        return out

    sweep_by_cos = _index(sweep, "sweep")
    base_by_cos = _index(linear_baseline, "linear_baseline")

    metrics: dict[str, float | int] = {"version": VERSION}

    method_sharp_vals = []
    baseline_sharp_vals = []

    for cos_AB in cos_sweep:
        key = round(cos_AB, 6)
        srec = sweep_by_cos.get(key, {})
        brec = base_by_cos.get(key, {})

        m_sharp = float(srec.get("and_sharpness", 0.0))
        fpr = float(srec.get("false_positive_rate", 1.0))
        fnr = float(srec.get("false_negative_rate", 1.0))
        b_sharp = float(brec.get("and_sharpness", 0.0))

        metrics[_cos_key("and_sharpness", cos_AB)] = m_sharp
        metrics[_cos_key("false_positive_rate", cos_AB)] = fpr
        metrics[_cos_key("false_negative_rate", cos_AB)] = fnr
        metrics[_cos_key("linear_baseline_sharpness", cos_AB)] = b_sharp

        method_sharp_vals.append(m_sharp)
        baseline_sharp_vals.append(b_sharp)

    # --- Canonical ---
    canonical_cos = float(payload["canonical_cosine"])
    canonical_key = _cos_key("and_sharpness", canonical_cos)
    metrics["and_sharpness_canonical"] = float(metrics.get(canonical_key, 0.0))

    baseline_canonical_key = _cos_key("linear_baseline_sharpness", canonical_cos)
    baseline_canonical = float(metrics.get(baseline_canonical_key, 0.0))
    metrics["lift_over_baseline_canonical"] = (
        metrics["and_sharpness_canonical"] - baseline_canonical
    )

    # --- Headline: superposition_robustness ---
    # Sharpness retained at max superposition (last cos) vs orthogonal (first cos).
    sharp_low = float(metrics.get(_cos_key("and_sharpness", cos_sweep[0]), 0.0))
    sharp_high = float(metrics.get(_cos_key("and_sharpness", cos_sweep[-1]), 0.0))
    if sharp_low > 1e-12:
        robustness = sharp_high / sharp_low
    else:
        robustness = 0.0
    metrics["superposition_robustness"] = float(max(0.0, min(1.0, robustness)))

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """
    Pipeline hook: True if metrics are clearly degenerate, to skip the jury.
    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # `and_sharpness` is clipped to [0, 1] (see task._sharpness), so a
    # multiplicative gate like `sharp > 1.5 * baseline` is unsatisfiable once
    # the baseline exceeds ~0.67 — it would fail even a perfect attempt. The
    # mechanically-degenerate case the jury should skip is an attempt that does
    # not even beat the no-mechanism linear baseline at the canonical condition.
    sharp = metrics.get("and_sharpness_canonical")
    baseline = metrics.get("linear_baseline_sharpness_cos_0p0")
    if isinstance(sharp, (int, float)) and isinstance(baseline, (int, float)):
        if baseline > 0 and sharp <= baseline:
            return True
        if baseline <= 0 and sharp <= 0:
            return True

    return False
