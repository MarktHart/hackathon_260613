import math

VERSION = 1

# Pipeline-only hook: how many GPU slots the attempt subprocess needs.
GPU_REQUIREMENT = 1


def _n_key(prefix: str, n: int) -> str:
    """Slice key name, e.g. ('distance_accuracy', 16) -> 'distance_accuracy_n_16'."""
    return f"{prefix}_n_{int(n)}"


def score(payload: dict) -> dict[str, float | int]:
    """Compute flat scalar metrics from the payload returned by task.evaluate().

    Expected payload keys:
        version (int == 1), canonical_n (int), n_nodes_sweep (list[int]),
        rel_tol (float), sweep (list[record]), linear_baseline (list[record]).

    Each sweep record:          {n_nodes, distance_accuracy, order_correlation, n_seeds}
    Each linear_baseline record:{n_nodes, distance_accuracy, order_correlation, n_seeds}
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    required = ["version", "canonical_n", "n_nodes_sweep", "sweep", "linear_baseline"]
    for k in required:
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    n_sweep = payload["n_nodes_sweep"]
    if not isinstance(n_sweep, (list, tuple)) or len(n_sweep) == 0:
        raise ValueError("payload['n_nodes_sweep'] must be a non-empty list")
    n_sweep = [int(n) for n in n_sweep]

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)) or len(sweep) != len(n_sweep):
        raise ValueError("payload['sweep'] must be a list of same length as n_nodes_sweep")

    baseline = payload["linear_baseline"]
    if not isinstance(baseline, (list, tuple)) or len(baseline) != len(n_sweep):
        raise ValueError(
            "payload['linear_baseline'] must be a list of same length as n_nodes_sweep"
        )

    def _index(records, what):
        out = {}
        for rec in records:
            if not isinstance(rec, dict):
                raise ValueError(f"Each {what} record must be a dict")
            if "n_nodes" not in rec:
                raise KeyError(f"{what} record missing 'n_nodes'")
            out[int(rec["n_nodes"])] = rec
        return out

    sweep_by_n = _index(sweep, "sweep")
    base_by_n = _index(baseline, "linear_baseline")

    metrics: dict[str, float | int] = {"version": VERSION}

    for n in n_sweep:
        srec = sweep_by_n.get(n, {})
        brec = base_by_n.get(n, {})

        acc = float(srec.get("distance_accuracy", 0.0))
        corr = float(srec.get("order_correlation", 0.0))
        b_acc = float(brec.get("distance_accuracy", 0.0))

        metrics[_n_key("distance_accuracy", n)] = acc
        metrics[_n_key("order_correlation", n)] = corr
        metrics[_n_key("onehop_baseline_accuracy", n)] = b_acc

    # --- Canonical condition ---
    canonical_n = int(payload["canonical_n"])
    metrics["distance_accuracy_canonical"] = float(
        metrics.get(_n_key("distance_accuracy", canonical_n), 0.0)
    )
    metrics["order_correlation_canonical"] = float(
        metrics.get(_n_key("order_correlation", canonical_n), 0.0)
    )
    base_canonical = float(metrics.get(_n_key("onehop_baseline_accuracy", canonical_n), 0.0))
    metrics["lift_over_baseline_canonical"] = (
        metrics["distance_accuracy_canonical"] - base_canonical
    )

    # --- Headline: dijkstra_robustness ---
    # How well distance accuracy is retained from the smallest (easiest) graph
    # to the largest (hardest, most relaxation hops) graph. A mechanism that
    # truly implements iterative relaxation degrades little with size.
    acc_small = float(metrics.get(_n_key("distance_accuracy", n_sweep[0]), 0.0))
    acc_large = float(metrics.get(_n_key("distance_accuracy", n_sweep[-1]), 0.0))
    if acc_small > 1e-12:
        robustness = acc_large / acc_small
    else:
        robustness = 0.0
    metrics["dijkstra_robustness"] = float(max(0.0, min(1.0, robustness)))

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Pipeline hook: True if metrics are clearly degenerate, to skip the jury.

    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # An attempt that cannot even match the no-propagation one-hop baseline at
    # the canonical condition has not learned any path-finding mechanism.
    acc = metrics.get("distance_accuracy_canonical")
    base = metrics.get(f"onehop_baseline_accuracy_n_16")
    if isinstance(acc, (int, float)) and isinstance(base, (int, float)):
        if acc <= base:
            return True

    return False
