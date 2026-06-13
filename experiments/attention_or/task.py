import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Batch:
    """Container for the canonical (orthogonal-anchor) synthetic problem instance."""
    K: np.ndarray          # (d, n) key matrix at the canonical condition
    q_A: np.ndarray        # (d,) unit query direction for feature A
    q_B: np.ndarray        # (d,) unit query direction for feature B
    cos_values: np.ndarray # (11,) sweep values cos(q_A, q_B) in [0, 1]


def _unit(v: np.ndarray) -> np.ndarray:
    """Normalize to unit norm; leave a zero vector unchanged."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _make_directions(d: int, cos_AB: float, rng: np.random.Generator):
    """Two unit query vectors q_A, q_B with exactly cos(q_A, q_B) == cos_AB."""
    A = _unit(rng.standard_normal(d))
    ortho = rng.standard_normal(d)
    ortho = ortho - np.dot(ortho, A) * A      # component orthogonal to A
    ortho = _unit(ortho)
    sin_AB = np.sqrt(max(0.0, 1.0 - cos_AB * cos_AB))
    B = cos_AB * A + sin_AB * ortho           # unit norm by construction
    return A, B


def _build_keys(A: np.ndarray, B: np.ndarray, d: int, n: int,
                rng: np.random.Generator) -> np.ndarray:
    """Key matrix: column 0 = k_A (=q_A dir), column 1 = k_B (=q_B dir),
    remaining n-2 columns = random unit noise keys."""
    K = rng.standard_normal((d, n))
    K = K / np.linalg.norm(K, axis=0, keepdims=True)
    K[:, 0] = A
    K[:, 1] = B
    return K


def _balanced_superposition(q_A: np.ndarray, q_B: np.ndarray) -> np.ndarray:
    """q_AB = normalize(q_A + q_B): the single combined query an OR mechanism
    must make attend to *both* signal keys at once."""
    return _unit(q_A + q_B)


def _build_problem(seed: int = 0) -> Batch:
    """Deterministic construction of the canonical (orthogonal) problem instance.

    Same seed -> identical Batch. The canonical anchor is cos(q_A, q_B) = 0.0,
    i.e. the first sweep value.
    """
    rng = np.random.default_rng(seed)
    d, n = 32, 64
    cos_values = np.linspace(0.0, 1.0, 11)
    A, B = _make_directions(d, float(cos_values[0]), rng)  # canonical: cos = 0.0
    K = _build_keys(A, B, d, n, rng)
    return Batch(K=K, q_A=A, q_B=B, cos_values=cos_values)


def generate(seed: int = 0) -> Batch:
    """Deterministic for a given seed. Same seed -> identical Batch."""
    return _build_problem(seed)


def evaluate(model_fn) -> dict:
    """
    Run `model_fn` over (q_A, q_B, q_AB) at each sweep value and return the
    payload dict exactly as benchmark.score expects it.

    The problem instance is rebuilt at each sweep point so that the swept
    quantity is the *actual* cosine similarity cos(q_A, q_B); the combined
    query is always the balanced superposition normalize(q_A + q_B).
    """
    d, n = 32, 64
    cos_values = np.linspace(0.0, 1.0, 11)

    # Deterministic, independent per-slice seeds derived from the canonical seed.
    master_rng = np.random.default_rng(0)
    slice_seeds = master_rng.integers(0, 2**31 - 1, size=len(cos_values))

    sweep_records = []
    for cos, sseed in zip(cos_values, slice_seeds):
        rng = np.random.default_rng(int(sseed))
        q_A, q_B = _make_directions(d, float(cos), rng)
        K = _build_keys(q_A, q_B, d, n, rng)
        q_AB = _balanced_superposition(q_A, q_B)

        s_A  = np.asarray(model_fn(q_A,  K), dtype=float).reshape(-1)
        s_B  = np.asarray(model_fn(q_B,  K), dtype=float).reshape(-1)
        s_AB = np.asarray(model_fn(q_AB, K), dtype=float).reshape(-1)

        # Indices 0 and 1 are the signal keys k_A, k_B; the rest are noise.
        sweep_records.append({
            "cos": float(cos),
            "s_A_at_A": float(s_A[0]),
            "s_A_at_B": float(s_A[1]),
            "s_A_noise_max": float(np.max(s_A[2:])) if len(s_A) > 2 else 0.0,
            "s_B_at_A": float(s_B[0]),
            "s_B_at_B": float(s_B[1]),
            "s_B_noise_max": float(np.max(s_B[2:])) if len(s_B) > 2 else 0.0,
            "s_AB_at_A": float(s_AB[0]),
            "s_AB_at_B": float(s_AB[1]),
            "s_AB_noise_max": float(np.max(s_AB[2:])) if len(s_AB) > 2 else 0.0,
        })

    return {
        "version": 1,
        "canonical_cos": 0.0,
        "sweep": sweep_records,
        "model_config": {
            "d": d,
            "n": n,
            "seed": 0,
        },
    }


def random_model_fn():
    """Returns a compliant model_fn that emits zeros (for smoke testing)."""
    def _fn(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        # query: (d,), keys: (d, n) -> returns (n,) zeros
        n = keys.shape[1]
        return np.zeros(n, dtype=np.float32)
    return _fn
