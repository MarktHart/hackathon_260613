import math
from typing import Dict, Any

VERSION = 1

# Expected sweep length (must match task.py)
EXPECTED_SWEEP_LEN = 5
CANONICAL_TOKEN = 128
L = 16  # sequence length (must match task.py)
# Expected fidelity of a no-mechanism uniform-attention head: its output is the
# mean of L i.i.d. value vectors, whose cosine to any single value is ~1/sqrt(L).
LINEAR_BASELINE = 1.0 / math.sqrt(L)  # ≈ 0.25 for L=16

def score(payload: Dict[str, Any]) -> Dict[str, float | int]:
    # Validate payload structure
    if "version" not in payload:
        raise KeyError("payload missing 'version'")
    if payload["version"] != 1:
        raise ValueError(f"Unsupported payload version {payload['version']}, expected 1")
    if "sweep" not in payload:
        raise KeyError("payload missing 'sweep'")
    if "canonical_token" not in payload:
        raise KeyError("payload missing 'canonical_token'")
    if "config" not in payload:
        raise KeyError("payload missing 'config'")

    sweep = payload["sweep"]
    if len(sweep) != EXPECTED_SWEEP_LEN:
        raise ValueError(f"sweep must have length {EXPECTED_SWEEP_LEN}, got {len(sweep)}")

    canonical_token = payload["canonical_token"]
    if canonical_token != CANONICAL_TOKEN:
        raise ValueError(f"canonical_token must be {CANONICAL_TOKEN}, got {canonical_token}")

    # Extract per-token metrics
    metrics: Dict[str, float | int] = {"version": VERSION}

    canonical_fidelity = None
    for record in sweep:
        token = record["token"]
        fid = record["copy_fidelity"]
        diag = record["diag_attn_mass"]

        metrics[f"identity_copy_fidelity_token_{token}"] = float(fid)
        metrics[f"diag_attn_mass_token_{token}"] = float(diag)

        if token == canonical_token:
            canonical_fidelity = fid

    if canonical_fidelity is None:
        raise ValueError(f"Canonical token {canonical_token} not found in sweep")

    metrics["identity_copy_fidelity_canonical"] = float(canonical_fidelity)
    metrics["linear_baseline_fidelity_canonical"] = LINEAR_BASELINE
    metrics["lift_over_linear_baseline"] = float(canonical_fidelity - LINEAR_BASELINE)

    return metrics

def is_obviously_broken(metrics: Dict[str, float | int]) -> bool:
    # NaN/inf check
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    # Fidelity worse than or equal to baseline (within small tolerance)
    fid = metrics.get("identity_copy_fidelity_canonical")
    base = metrics.get("linear_baseline_fidelity_canonical")
    if isinstance(fid, float | int) and isinstance(base, float | int):
        if fid <= base * 1.1:  # at/just-above the uniform-attention floor → no real mechanism
            return True
    return False

GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU
