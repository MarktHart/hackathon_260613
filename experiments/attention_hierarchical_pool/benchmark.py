import math
from typing import Any

VERSION = 1

# Baseline: uniform attention within the query's OWN chunk (16 tokens).
# Under this "no-mechanism" strawman each query spreads mass 1/C over the C
# keys of its chunk and 0 elsewhere. The resulting concentrations, measured
# exactly the way task._compute_concentrations measures them, are:
# - chunk (whole chunk): all mass lands in the query's chunk -> 1.0
# - superchunk (4 chunks): the query's chunk is fully contained in its
#   superchunk, so all mass is inside it too -> 1.0 (NOT C/(4C); that would be
#   the value for uniform-within-superchunk, a different pattern)
# - local (same-chunk keys within ±2 of the query): the ±2 window is trimmed at
#   chunk edges, so averaged over the C query positions in a chunk it holds
#   (3 + 4 + 5*(C-4) + 4 + 3)/C keys. For C=16 that is 74/16 = 4.625 keys, each
#   of weight 1/16 -> 74/256 = 0.2890625 (NOT 5/16, which ignores the trim)
# - entropy: uniform over C keys -> log(C) = log(16) ≈ 2.77 nats
_CHUNK_SIZE = 16
_BASELINE_LOCAL = 74.0 / 256.0
_BASELINE_CHUNK = 1.0
_BASELINE_SUPERCHUNK = 1.0
_BASELINE_ENTROPY = math.log(_CHUNK_SIZE)


def score(payload: dict) -> dict[str, float | int]:
    """Compute hierarchical pooling metrics from task payload."""
    # Validate payload
    required_keys = {"version", "seq_len", "num_layers", "num_heads", "chunk_size", "num_chunks", "sweep"}
    missing = required_keys - set(payload.keys())
    if missing:
        raise ValueError(f"Payload missing keys: {missing}")

    if payload["version"] != 1:
        raise ValueError(f"Unsupported payload version: {payload['version']}")

    sweep = payload["sweep"]
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("Payload 'sweep' must be a non-empty list")

    num_layers = payload["num_layers"]
    num_heads = payload["num_heads"]
    expected_len = num_layers * num_heads
    if len(sweep) != expected_len:
        raise ValueError(f"Sweep length {len(sweep)} != num_layers * num_heads ({expected_len})")

    # Validate each record
    for i, rec in enumerate(sweep):
        for k in ("layer", "head", "local_concentration", "chunk_concentration",
                  "superchunk_concentration", "entropy"):
            if k not in rec:
                raise ValueError(f"Sweep record {i} missing key: {k}")
        if not (0 <= rec["layer"] < num_layers):
            raise ValueError(f"Record {i}: layer {rec['layer']} out of range [0, {num_layers})")
        if not (0 <= rec["head"] < num_heads):
            raise ValueError(f"Record {i}: head {rec['head']} out of range [0, {num_heads})")

    # Group by layer
    by_layer: dict[int, list[dict]] = {L: [] for L in range(num_layers)}
    for rec in sweep:
        by_layer[rec["layer"]].append(rec)

    metrics: dict[str, float | int] = {"version": VERSION}

    # Per-layer medians
    for layer in range(num_layers):
        recs = by_layer[layer]
        n_heads = len(recs)
        if n_heads == 0:
            continue

        local_med = median([r["local_concentration"] for r in recs])
        chunk_med = median([r["chunk_concentration"] for r in recs])
        superchunk_med = median([r["superchunk_concentration"] for r in recs])
        entropy_med = median([r["entropy"] for r in recs])

        metrics[f"local_concentration_layer_{layer}"] = local_med
        metrics[f"chunk_concentration_layer_{layer}"] = chunk_med
        metrics[f"superchunk_concentration_layer_{layer}"] = superchunk_med
        metrics[f"entropy_layer_{layer}"] = entropy_med

        # Baselines (same for every layer since baseline is layer-agnostic)
        metrics[f"linear_baseline_local_concentration_layer_{layer}"] = _BASELINE_LOCAL
        metrics[f"linear_baseline_chunk_concentration_layer_{layer}"] = _BASELINE_CHUNK
        metrics[f"linear_baseline_superchunk_concentration_layer_{layer}"] = _BASELINE_SUPERCHUNK
        metrics[f"linear_baseline_entropy_layer_{layer}"] = _BASELINE_ENTROPY

    # Headline: hierarchical_robustness_canonical
    # Measures a fine -> coarse pooling SHIFT with depth, normalised so that
    # depth-invariant attention (uniform, or any pattern whose spread does not
    # change with layer) scores ~1.0 regardless of absolute region sizes.
    #
    # For each layer define the within-chunk spread
    #     spread_L = chunk_concentration_layer_L / local_concentration_layer_L
    # i.e. how far mass spreads across the whole chunk relative to the tight ±2
    # local window. The headline is
    #     median_{late layers}(spread) / median_{early layers}(spread).
    # > 1  => late layers pool coarsely (mass spread across the chunk) while
    #         early layers stay locally concentrated -- the hierarchical
    #         signature. Comparing spread-to-spread cancels the region-size
    #         offset, so uniform attention scores 1.0, not ~3.5.
    early_layers = list(range(0, num_layers // 2))
    late_layers = list(range(num_layers // 2, num_layers))

    def _spread(L: int) -> float | None:
        lk = f"local_concentration_layer_{L}"
        ck = f"chunk_concentration_layer_{L}"
        if lk not in metrics or ck not in metrics:
            return None
        local_v = metrics[lk]
        if local_v <= 0:
            return None
        return metrics[ck] / local_v

    early_spread = [s for L in early_layers if (s := _spread(L)) is not None]
    late_spread = [s for L in late_layers if (s := _spread(L)) is not None]

    if early_spread and late_spread:
        early_med = median(early_spread)
        late_med = median(late_spread)
        if early_med > 0:
            metrics["hierarchical_robustness_canonical"] = late_med / early_med
        else:
            metrics["hierarchical_robustness_canonical"] = float('nan')
    else:
        metrics["hierarchical_robustness_canonical"] = float('nan')

    return metrics


def median(values: list[float]) -> float:
    """Return median of a non-empty list."""
    if not values:
        return float('nan')
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return sorted_vals[mid]


def is_obviously_broken(metrics: dict) -> bool:
    """Pipeline hook: return True if metrics indicate a catastrophically failed run."""
    # NaN/inf check
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # Headline metric worse than or equal to 1.0 means no hierarchical shift
    # (late-layer chunk pooling <= early-layer local pooling)
    robustness = metrics.get("hierarchical_robustness_canonical")
    if isinstance(robustness, float | int) and robustness <= 1.0:
        return True

    # Entropy higher than baseline in all layers = attention never sharpens
    entropies = [v for k, v in metrics.items() if k.startswith("entropy_layer_")]
    baselines = [v for k, v in metrics.items() if k.startswith("linear_baseline_entropy_layer_")]
    if entropies and baselines and all(e >= b for e, b in zip(entropies, baselines)):
        return True

    return False

GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU
