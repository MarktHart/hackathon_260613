import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple

# model_fn contract: tokens [batch, 3] int -> (Q, K) each [batch, 3, d_head] float
ModelFn = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]

P = 97          # canonical prime modulus
D_HEAD = 128    # canonical head dimension


@dataclass(frozen=True)
class Batch:
    tokens: np.ndarray          # [batch, 3] int32  columns = [a, b, p]
    a_vals: np.ndarray          # [batch] int32
    b_vals: np.ndarray          # [batch] int32
    p: int
    d_head: int


def generate(seed: int = 0) -> Batch:
    """
    Generate all pairs (a, b) for modular addition with prime modulus p.

    The canonical condition is fully fixed (p, d_head, full Cartesian product),
    so `seed` is accepted but ignored — same seed (any seed) -> same batch.
    """
    p = P
    a_vals = np.repeat(np.arange(p), p)               # [p^2]
    b_vals = np.tile(np.arange(p), p)                 # [p^2]
    eq = np.full(p * p, p, dtype=np.int32)            # '=' separator token == p
    tokens = np.stack([a_vals, b_vals, eq], axis=1)   # [p^2, 3]

    return Batch(
        tokens=tokens.astype(np.int32),
        a_vals=a_vals.astype(np.int32),
        b_vals=b_vals.astype(np.int32),
        p=p,
        d_head=D_HEAD,
    )


def _fourier_features(p: int) -> np.ndarray:
    """
    Fourier features for every token value x in [0, p).
    Returns [p, 2*(p//2)]; columns are (sin_k, cos_k) for k = 1..p//2.
    """
    n_freqs = p // 2
    features = np.zeros((p, 2 * n_freqs), dtype=np.float64)
    x = np.arange(p, dtype=np.float64)
    for k in range(1, n_freqs + 1):
        features[:, 2 * (k - 1)] = np.sin(2 * np.pi * k * x / p)
        features[:, 2 * (k - 1) + 1] = np.cos(2 * np.pi * k * x / p)
    return features


def _clip(v: float, lo: float, hi: float) -> float:
    if not np.isfinite(v):
        return lo
    return float(min(max(v, lo), hi))


def _compute_sweep(
    Q: np.ndarray,           # [batch, d_head]
    K: np.ndarray,           # [batch, d_head]
    a_vals: np.ndarray,      # [batch]
    b_vals: np.ndarray,      # [batch]
    p: int,
) -> list:
    """
    For each Fourier frequency k compute:
      - alignment: mean cosine of principal angles between the head's Q-side and
        K-side freq-k subspaces (in [0, 1]);
      - phase_error: magnitude-weighted deviation from the conjugate-phase
        prediction w_K = conj(w_Q) (in [0, pi]);
      - explained_variance: fraction of centered ||Q||^2 + ||K||^2 captured by
        the freq-k projection (in [0, 1]).
    """
    n_freqs = p // 2
    batch_size = Q.shape[0]

    features = _fourier_features(p)          # [p, 2*n_freqs]

    Q_c = (Q - Q.mean(axis=0, keepdims=True)).astype(np.float64)
    K_c = (K - K.mean(axis=0, keepdims=True)).astype(np.float64)

    total_var = float(np.sum(Q_c ** 2) + np.sum(K_c ** 2)) + 1e-8

    results = []
    for k in range(1, n_freqs + 1):
        feat_a = features[a_vals][:, 2 * (k - 1):2 * k]   # [batch, 2]
        feat_b = features[b_vals][:, 2 * (k - 1):2 * k]   # [batch, 2]

        # Regress Q/K onto the (orthogonal) freq-k feature pair.
        W_Q = feat_a.T @ Q_c / batch_size                # [2, d_head]
        W_K = feat_b.T @ K_c / batch_size                # [2, d_head]

        # Explained variance via the freq-k least-squares reconstruction. The
        # sin/cos columns are orthogonal over the full (a, b) grid, so the LS
        # coefficient for each column is (featᵀ x) / ||feat||². Normalising by
        # the column energy (not batch_size) makes explained_var a true
        # *fraction* of centered variance; using batch_size would scale it by
        # ||feat||²/batch ≈ 1/2 and the squared norm by ≈ 1/4. (Scaling W_Q/W_K
        # this way would not change alignment or phase_error, so those keep
        # using the batch-normalised W_Q/W_K above.)
        col_sq_a = np.sum(feat_a ** 2, axis=0) + 1e-12   # [2]
        col_sq_b = np.sum(feat_b ** 2, axis=0) + 1e-12   # [2]
        Q_proj = feat_a @ ((feat_a.T @ Q_c) / col_sq_a[:, None])
        K_proj = feat_b @ ((feat_b.T @ K_c) / col_sq_b[:, None])
        explained_var = (float(np.sum(Q_proj ** 2) + np.sum(K_proj ** 2))) / total_var

        # Alignment: cosines of the principal angles between the two freq-k
        # subspaces in R^{d_head}. The row space of W_Q (resp. W_K) is spanned
        # by the *right* singular vectors Vt (each row a unit vector in
        # R^{d_head}); the singular values of Vt_Q @ Vt_K.T are those cosines.
        # (Using the left vectors U would give a 2x2 orthogonal product whose
        # singular values are trivially 1 — a degenerate metric.)
        _, _, Vt_Q = np.linalg.svd(W_Q, full_matrices=False)  # Vt_Q [2, d_head]
        _, _, Vt_K = np.linalg.svd(W_K, full_matrices=False)  # Vt_K [2, d_head]
        cos_angles = np.linalg.svd(Vt_Q @ Vt_K.T, compute_uv=False)
        alignment = float(np.mean(cos_angles))

        # Phase error: treat the two rows as a complex direction per channel.
        w_Q = W_Q[0] + 1j * W_Q[1]                        # [d_head]
        w_K = W_K[0] + 1j * W_K[1]                        # [d_head]
        phase_diffs = np.angle(w_Q * np.conj(w_K))        # [d_head] in (-pi, pi]
        mags = np.abs(w_Q) * np.abs(w_K)
        if mags.sum() > 1e-12:
            phase_error = float(np.average(np.abs(phase_diffs), weights=mags))
        else:
            phase_error = float(np.pi)

        results.append({
            "frequency": int(k),
            "alignment": _clip(alignment, 0.0, 1.0),
            "phase_error": _clip(phase_error, 0.0, float(np.pi)),
            "explained_variance": _clip(explained_var, 0.0, 1.0),
        })

    return results


def evaluate(model_fn: ModelFn) -> dict:
    """
    Run `model_fn` over the full modular-addition batch and return the payload
    consumed by benchmark.score(). Attempts never build the payload themselves.
    """
    batch = generate(seed=0)
    p = batch.p
    d_head = batch.d_head

    Q_all, K_all = model_fn(batch.tokens)
    Q_all = np.asarray(Q_all)
    K_all = np.asarray(K_all)

    expected = (p * p, 3, d_head)
    if Q_all.shape != expected or K_all.shape != expected:
        raise ValueError(
            f"model_fn returned Q/K of shape {Q_all.shape}/{K_all.shape}, "
            f"expected {expected}"
        )

    Q_a = Q_all[:, 0, :]   # query at the 'a' position
    K_b = K_all[:, 1, :]   # key at the 'b' position

    sweep = _compute_sweep(Q_a, K_b, batch.a_vals, batch.b_vals, p)

    alignments = [s["alignment"] for s in sweep]
    max_alignment = max(alignments)
    argmax_freq = sweep[alignments.index(max_alignment)]["frequency"]
    total_explained = float(sum(s["explained_variance"] for s in sweep))

    return {
        "version": 1,
        "modulus": int(p),
        "d_head": int(d_head),
        "layer_index": 0,
        "head_index": 0,
        "sweep": sweep,
        "total_explained_variance": total_explained,
        "max_alignment": float(max_alignment),
        "argmax_alignment_freq": int(argmax_freq),
    }


def random_model_fn() -> ModelFn:
    """
    A contract-shaped model_fn that returns Gaussian-noise Q, K of the right
    shape. Pure NumPy. Deterministic given the token batch (seeded off the
    token sum) so the smoke test is reproducible.
    """
    def _fn(tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tokens = np.asarray(tokens)
        batch_size, seq_len = tokens.shape
        d_head = D_HEAD
        seed = int(tokens.sum()) % (2 ** 32)
        rng = np.random.default_rng(seed)
        Q = rng.normal(0.0, 1.0, size=(batch_size, seq_len, d_head)).astype(np.float32)
        K = rng.normal(0.0, 1.0, size=(batch_size, seq_len, d_head)).astype(np.float32)
        return Q, K
    return _fn
