import numpy as np
from dataclasses import dataclass
from typing import Callable

# ModelFn signature as documented in README.md
ModelFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

# Canonical measurement condition (see README.md).
D = 64
N_POSITIONS = 100
FEATURE_DENSITY = 0.3
CANONICAL_COSINE = 0.0
COS_AB_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_SEEDS = 10
EVAL_SEED = 42


@dataclass(frozen=True)
class Batch:
    """One evaluation batch: multiple (q_A, q_B, residual, labels) tuples across seeds."""
    q_As: list[np.ndarray]          # list of (d,) per (cos, seed)
    q_Bs: list[np.ndarray]          # list of (d,) per (cos, seed)
    residuals: list[np.ndarray]     # list of (n_positions, d) per (cos, seed)
    labels: list[np.ndarray]        # list of (n_positions,) bool per (cos, seed) — ground truth A∧B
    cosines: list[float]            # nominal cosine for each entry


def generate(seed: int = 0) -> Batch:
    """
    Deterministic batch for the canonical sweep.

    Returns len(COS_AB_SWEEP) * N_SEEDS evaluation conditions. `seed` shifts the
    per-entry RNG so the whole batch is reproducible for a given seed.
    """
    d = D
    n_positions = N_POSITIONS
    feature_density = FEATURE_DENSITY

    q_As: list[np.ndarray] = []
    q_Bs: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    cosines: list[float] = []

    for ci, cos_val in enumerate(COS_AB_SWEEP):
        for s in range(N_SEEDS):
            # Deterministic, well-distributed per-entry seed.
            seed_i = (np.uint64(seed) * np.uint64(1_000_003)
                      + np.uint64(ci) * np.uint64(9_973)
                      + np.uint64(s)) & np.uint64(0xFFFFFFFF)
            r = np.random.default_rng(int(seed_i))

            # Sample q_A uniformly on the sphere.
            q_A = r.normal(size=d)
            q_A = q_A / np.linalg.norm(q_A)

            # Construct q_B with controlled cosine to q_A:
            #   q_B = cos * q_A + sqrt(1 - cos^2) * ortho
            ortho = r.normal(size=d)
            ortho = ortho - np.dot(ortho, q_A) * q_A
            ortho_norm = np.linalg.norm(ortho)
            if ortho_norm < 1e-10:
                ortho = np.zeros(d)
            else:
                ortho = ortho / ortho_norm
            q_B = cos_val * q_A + np.sqrt(max(0.0, 1.0 - cos_val ** 2)) * ortho
            q_B = q_B / np.linalg.norm(q_B)

            # Residual stream: noise + feature contributions.
            residual = r.normal(size=(n_positions, d)) * 0.5

            feat_A = r.random(n_positions) < feature_density
            feat_B = r.random(n_positions) < feature_density

            residual[feat_A] += q_A * 2.0
            residual[feat_B] += q_B * 2.0

            label = feat_A & feat_B  # ground-truth AND

            q_As.append(q_A.astype(np.float32))
            q_Bs.append(q_B.astype(np.float32))
            residuals.append(residual.astype(np.float32))
            labels.append(label.astype(bool))
            cosines.append(float(cos_val))

    return Batch(
        q_As=q_As,
        q_Bs=q_Bs,
        residuals=residuals,
        labels=labels,
        cosines=cosines,
    )


def _sharpness(scores: np.ndarray, label: np.ndarray) -> float:
    """Normalised separation of scores on AND positions vs the rest, in [0, 1]."""
    if np.any(label) and np.any(~label):
        mean_on = float(scores[label].mean())
        mean_off = float(scores[~label].mean())
        denom = max(abs(mean_on), 1e-8)
        return max(0.0, min(1.0, (mean_on - mean_off) / denom))
    return 0.0


def evaluate(model_fn: ModelFn) -> dict:
    """
    Run `model_fn` over the canonical batch and return a payload dict matching
    the benchmark.score contract exactly.
    """
    batch = generate(seed=EVAL_SEED)

    # Group attempt results by nominal cosine.
    by_cosine: dict[float, list[dict]] = {c: [] for c in COS_AB_SWEEP}
    # Group baseline (no-mechanism) results by nominal cosine.
    base_by_cosine: dict[float, list[float]] = {c: [] for c in COS_AB_SWEEP}

    for q_A, q_B, residual, label, cos_val in zip(
        batch.q_As, batch.q_Bs, batch.residuals, batch.labels, batch.cosines
    ):
        n_positions = residual.shape[0]

        # --- Attempt's model ---
        logits = np.asarray(model_fn(q_A, q_B, residual), dtype=np.float64).reshape(-1)
        if logits.shape != (n_positions,):
            raise ValueError(
                f"model_fn returned shape {logits.shape}, expected ({n_positions},)"
            )
        attn = np.exp(logits - logits.max())
        attn = attn / attn.sum()

        threshold = 1.0 / n_positions  # uniform attention level
        pred = attn > threshold

        tp = int(np.sum(pred & label))
        fp = int(np.sum(pred & ~label))
        fn = int(np.sum(~pred & label))
        tn = int(np.sum(~pred & ~label))

        by_cosine[cos_val].append({
            "and_sharpness": _sharpness(attn, label),
            "false_positive_rate": fp / max(fp + tn, 1),
            "false_negative_rate": fn / max(fn + tp, 1),
        })

        # --- Linear baseline: AND-like sum of the two probe projections ---
        x_A = residual @ q_A
        x_B = residual @ q_B
        score = x_A + x_B
        base_by_cosine[cos_val].append(_sharpness(score, label))

    # Aggregate across seeds.
    sweep = []
    linear_baseline = []
    for cos_val in COS_AB_SWEEP:
        records = by_cosine[cos_val]
        sweep.append({
            "cosine": float(cos_val),
            "and_sharpness": float(np.mean([r["and_sharpness"] for r in records])),
            "false_positive_rate": float(np.mean([r["false_positive_rate"] for r in records])),
            "false_negative_rate": float(np.mean([r["false_negative_rate"] for r in records])),
            "n_seeds": len(records),
        })
        base_vals = base_by_cosine[cos_val]
        linear_baseline.append({
            "cosine": float(cos_val),
            "and_sharpness": float(np.mean(base_vals)) if base_vals else 0.0,
            "n_seeds": len(base_vals),
        })

    return {
        "version": 2,
        "model_name": "synthetic_attention_and",
        "d": D,
        "canonical_cosine": CANONICAL_COSINE,
        "cos_AB_sweep": list(COS_AB_SWEEP),
        "sweep": sweep,
        "linear_baseline": linear_baseline,
    }


def random_model_fn() -> ModelFn:
    """
    Return a model_fn with the real signature whose body emits random logits.
    Pure NumPy; used by the pipeline smoke test.
    """
    rng = np.random.default_rng(0)

    def _random_fn(q_A: np.ndarray, q_B: np.ndarray, residual: np.ndarray) -> np.ndarray:
        n_positions = np.asarray(residual).shape[0]
        return rng.normal(size=n_positions).astype(np.float32)

    return _random_fn
