"""Benchmark for the `attention_connected_components` goal.

Pure Python, no numpy. Consumes the payload returned by `task.evaluate` and
emits a flat dict of named scalar metrics. Deterministic and side-effect free.

The question: does an attention matrix recover the *transitive closure* of a
graph (which nodes share a connected component) rather than only the 1-hop
adjacency relation? The attempt hands over a same-component affinity matrix;
we score the induced pairwise "same component?" relation against ground truth
with the F1 statistic, and contrast it with the adjacency-only baseline.
"""

import math

VERSION = 1

# Pipeline-only hook: how many GPU slots the attempt subprocess needs.
GPU_REQUIREMENT = 1

_COUNT_KEYS = ("tp", "fp", "fn", "tn")


def _diam_key(prefix: str, diameter) -> str:
    """Slice key name, e.g. ('cc_f1', 3) -> 'cc_f1_diam_3'. Diameters are ints."""
    return f"{prefix}_diam_{int(diameter)}"


def _f1(counts: dict) -> float:
    """F1 of the pairwise same-component relation from a confusion dict.

    F1 = 2*tp / (2*tp + fp + fn). Returns 0.0 when the denominator is zero
    (no positive predictions and no reachable positive truths), so the metric
    is always defined and bounded in [0, 1].
    """
    if not isinstance(counts, dict):
        raise ValueError("confusion record must be a dict")
    for k in _COUNT_KEYS:
        if k not in counts:
            raise KeyError(f"confusion record missing {k!r}")
    tp = float(counts["tp"])
    fp = float(counts["fp"])
    fn = float(counts["fn"])
    denom = 2.0 * tp + fp + fn
    if denom <= 0.0:
        return 0.0
    return (2.0 * tp) / denom


def score(payload: dict) -> dict[str, float | int]:
    """Compute flat scalar metrics from the payload returned by task.evaluate().

    Expected payload keys:
        version (int == 1)
        canonical_diameter (int)
        num_components (int)
        num_graphs (int)
        sweep (list[record]), non-empty

    Each sweep record:
        {"diameter": int,
         "model":    {"tp","fp","fn","tn"},
         "baseline": {"tp","fp","fn","tn"}}
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    for k in ("version", "canonical_diameter", "sweep"):
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)) or len(sweep) == 0:
        raise ValueError("payload['sweep'] must be a non-empty list")

    canonical = int(payload["canonical_diameter"])

    metrics: dict[str, float | int] = {"version": VERSION}

    model_f1s: list[float] = []
    lifts: list[float] = []
    canonical_model = None
    canonical_base = None

    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError("each sweep record must be a dict")
        for k in ("diameter", "model", "baseline"):
            if k not in rec:
                raise KeyError(f"sweep record missing {k!r}")
        diameter = int(rec["diameter"])

        m_f1 = _f1(rec["model"])
        b_f1 = _f1(rec["baseline"])
        lift = m_f1 - b_f1

        metrics[_diam_key("cc_f1", diameter)] = m_f1
        metrics[_diam_key("adjacency_baseline_f1", diameter)] = b_f1
        metrics[_diam_key("lift_over_adjacency", diameter)] = lift

        model_f1s.append(m_f1)
        lifts.append(lift)

        if diameter == canonical:
            canonical_model = m_f1
            canonical_base = b_f1

    # Fall back to the first slice if the canonical diameter is absent.
    if canonical_model is None:
        canonical_model = _f1(sweep[0]["model"])
        canonical_base = _f1(sweep[0]["baseline"])

    metrics["cc_f1_canonical"] = float(canonical_model)
    metrics["adjacency_baseline_f1_canonical"] = float(canonical_base)
    metrics["lift_over_adjacency_canonical"] = float(canonical_model - canonical_base)

    # Headline: mean pairwise F1 across the diameter sweep, bounded in [0, 1].
    metrics["transitive_closure_robustness"] = sum(model_f1s) / len(model_f1s)
    metrics["mean_lift_over_adjacency"] = sum(lifts) / len(lifts)

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Pipeline hook: True if metrics are clearly degenerate, to skip the jury.

    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # The whole point is to beat the 1-hop adjacency baseline at recovering the
    # transitive closure. An attempt that does not even match the adjacency
    # baseline at the canonical condition is mechanically degenerate.
    model = metrics.get("cc_f1_canonical")
    baseline = metrics.get("adjacency_baseline_f1_canonical")
    if isinstance(model, (int, float)) and isinstance(baseline, (int, float)):
        if model <= baseline:
            return True

    return False
