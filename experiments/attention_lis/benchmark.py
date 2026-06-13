import numpy as np
import math

VERSION = 1

# Canonical sweep noise stds (must match task.py)
SWEEP_NOISE_STDS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]


def _encoding_dirs(proj: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """
    Estimate the direction in projection-space along which the model encodes
    each factor: the difference of class means between factor = +1 and -1
    positions. W[k] is the encoding direction of factor k, living in R^K.

    proj:    (K, L) — projected queries/keys (q_proj rows are factor coords)
    factors: (L, K) — ground-truth factors in {-1, +1}
    returns: W (K, K)
    """
    K, L = proj.shape
    X = proj.T  # (L, K) — per-position projected representation
    W = np.zeros((K, K), dtype=np.float64)
    for k in range(K):
        pos = factors[:, k] > 0
        neg = ~pos
        if pos.any() and neg.any():
            W[k] = X[pos].mean(axis=0) - X[neg].mean(axis=0)
    return W


def _orthogonality(proj: np.ndarray, factors: np.ndarray) -> float:
    """
    Linear-independence of the K factor subspaces: 1 - |cosine| averaged over
    k≠k', where each direction is the model's *encoding direction* for that
    factor (difference-of-means in projection space). Orthogonal encoding
    directions ⇒ the factors occupy linearly independent subspaces (LIS ≈ 1).

    A near-zero-norm encoding direction is degenerate (e.g. a model that emits
    all-zero/constant q/k, or never separates a factor). It carries no
    orientation, so it must NOT be rewarded as "orthogonal" — otherwise a
    structureless model scores a perfect 1.0. Any pair touching a degenerate
    direction is treated as maximally aligned (|cos| = 1), driving its
    orthogonality contribution to 0.

    proj:    (K, L) — projected queries/keys
    factors: (L, K) — ground-truth factors in {-1, +1}
    """
    K = proj.shape[0]
    if K < 2:
        return 1.0
    W = _encoding_dirs(proj, factors)  # (K, K)
    norms = np.linalg.norm(W, axis=1)  # (K,)
    degenerate = norms < 1e-8
    safe_norms = np.maximum(norms, 1e-12)[:, None]
    normalized = W / safe_norms  # (K, K)
    cosines = normalized @ normalized.T  # (K, K)
    abs_cos = np.abs(cosines)
    deg_pair = degenerate[:, None] | degenerate[None, :]
    abs_cos = np.where(deg_pair, 1.0, abs_cos)
    mask = ~np.eye(K, dtype=bool)
    return float(1.0 - np.mean(abs_cos[mask]))


def _alignment(proj: np.ndarray, factors: np.ndarray) -> float:
    """
    proj: (K, L) — projected queries/keys
    factors: (L, K) — ground truth factors in {-1, +1}
    Returns mean correlation between each proj[k] and factors[:, k].
    """
    K = proj.shape[0]
    corrs = []
    for k in range(K):
        p = proj[k]  # (L,)
        f = factors[:, k]  # (L,)
        # Pearson correlation
        p_centered = p - np.mean(p)
        f_centered = f - np.mean(f)
        num = np.sum(p_centered * f_centered)
        den = np.sqrt(np.sum(p_centered**2) * np.sum(f_centered**2))
        if den > 1e-8:
            corrs.append(num / den)
        else:
            corrs.append(0.0)
    return float(np.mean(corrs))


def _linear_baseline_orthogonality(d_model: int, K: int, L: int, factors: np.ndarray, seed: int = 42) -> float:
    """
    Orthogonality of a no-mechanism model: random Gaussian q independent of the
    factors, scored under the identical encoding-direction metric. Because q
    carries no factor information, its encoding directions are random ⇒ low
    orthogonality. A real method must beat this.
    """
    rng = np.random.default_rng(seed)
    # Random q: (L, d_model), independent of factors
    q = rng.normal(size=(L, d_model)).astype(np.float32)
    # Random orthonormal factor_directions: (K, d_model)
    A = rng.normal(size=(K, d_model)).astype(np.float32)
    Q_dir, _ = np.linalg.qr(A.T, mode='reduced')
    factor_directions = Q_dir.T.astype(np.float32)
    # Project: (K, L)
    proj = (q @ factor_directions.T).T
    return _orthogonality(proj, factors)


def score(payload: dict) -> dict[str, float | int]:
    """
    Compute all LIS metrics from the payload.
    """
    # Validate required keys
    required_keys = ["version", "config", "canonical", "sweep", "factor_directions", "factors"]
    for k in required_keys:
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k}")

    canonical = payload["canonical"]
    sweep = payload["sweep"]
    factor_directions = payload["factor_directions"]
    factors = payload["factors"]
    config = payload["config"]

    K = config["K"]
    d_model = config["d_model"]
    seq_len = config["seq_len"]

    # Validate shapes
    if canonical["q_proj"].shape != (K, seq_len):
        raise ValueError(f"canonical q_proj shape {canonical['q_proj'].shape} != ({K}, {seq_len})")
    if canonical["k_proj"].shape != (K, seq_len):
        raise ValueError(f"canonical k_proj shape {canonical['k_proj'].shape} != ({K}, {seq_len})")
    if factor_directions.shape != (K, d_model):
        raise ValueError(f"factor_directions shape {factor_directions.shape} != ({K}, {d_model})")
    if factors.shape != (seq_len, K):
        raise ValueError(f"factors shape {factors.shape} != ({seq_len}, {K})")
    if len(sweep) != len(SWEEP_NOISE_STDS):
        raise ValueError(f"sweep length {len(sweep)} != {len(SWEEP_NOISE_STDS)}")

    metrics = {}
    metrics["version"] = VERSION

    # Canonical orthogonality (queries)
    ortho_q_canon = _orthogonality(canonical["q_proj"], factors)
    metrics["lis_orthogonality_canonical"] = ortho_q_canon

    # Canonical alignment (queries vs ground truth factors)
    align_q_canon = _alignment(canonical["q_proj"], factors)
    metrics["lis_alignment_canonical"] = align_q_canon

    # Sweep orthogonality and alignment
    ortho_sweep = []
    align_sweep = []
    for entry in sweep:
        ns = entry["noise_std"]
        q_proj = entry["q_proj"]
        if q_proj.shape != (K, seq_len):
            raise ValueError(f"sweep q_proj shape {q_proj.shape} != ({K}, {seq_len}) at noise_std={ns}")
        ortho = _orthogonality(q_proj, factors)
        align = _alignment(q_proj, factors)
        ortho_sweep.append(ortho)
        align_sweep.append(align)
        # Per-slice metrics with 0p7 formatting
        ns_str = f"{ns:.1f}".replace(".", "p")
        metrics[f"lis_orthogonality_noise_{ns_str}"] = ortho
        metrics[f"lis_alignment_noise_{ns_str}"] = align

    # Robustness: min orthogonality across sweep / orthogonality at noise=0
    ortho_noise_0 = ortho_sweep[0]  # noise_std=0.0 is first
    if ortho_noise_0 > 1e-8:
        metrics["lis_robustness"] = float(min(ortho_sweep) / ortho_noise_0)
    else:
        metrics["lis_robustness"] = 0.0

    # Linear baseline (random model)
    baseline_ortho = _linear_baseline_orthogonality(d_model, K, seq_len, factors)
    metrics["linear_baseline_orthogonality_canonical"] = baseline_ortho
    metrics["lift_over_linear_baseline_canonical"] = ortho_q_canon - baseline_ortho

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """
    Detect clearly degenerate results that shouldn't go to the jury.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # If orthogonality is worse than or equal to linear baseline, it's broken
    ortho = metrics.get("lis_orthogonality_canonical")
    baseline = metrics.get("linear_baseline_orthogonality_canonical")
    if isinstance(ortho, (int, float)) and isinstance(baseline, (int, float)):
        if ortho <= baseline * 1.01:  # allow tiny numerical noise
            return True

    # Robustness should be in [0, 1]
    robustness = metrics.get("lis_robustness")
    if isinstance(robustness, float):
        if robustness < 0.0 or robustness > 1.0:
            return True

    return False