"""Task definition for the `attention_and` goal.

The goal owns this file just like it owns ``benchmark.py``. Every attempt
imports it so the cosine sweep, the canonical ``d`` / ``noise_scale`` / ``seed``,
the feature/query geometry, and the residual-stream construction are
byte-identical across attempts — no drift in what the question actually is.

An attempt only contributes a **model function** with the signature documented
in ``README.md``::

    def model_fn(q_A, q_B, x) -> tuple[float, float]:
        # q_A, q_B : np.ndarray, shape (d,)  -- query vectors for features A, B
        # x        : np.ndarray, shape (d,)  -- a residual-stream vector
        # returns  : (logit_A, logit_B)      -- the head's scalar response of
        #                                       each query to this residual stream

``evaluate`` constructs, for every cosine ``c`` in :data:`COSINE_SWEEP`, the
fixed orthogonal feature directions and the two query vectors (each at cosine
``c`` with its own feature, orthogonal to the other feature and to each other),
then probes the model on the four presence configurations
``(alpha, beta) in {(1,1), (1,0), (0,1), (0,0)}``. For stability each reported
logit is the mean of the model's response over :data:`N_NOISE_SAMPLES`
independent noise draws (the payload still reports a single float per quantity,
exactly as the README contract specifies). It assembles the dict that
``benchmark.score`` consumes — pass it straight to ``record_benchmark``.

Pure Python + NumPy; no I/O, no network. Deterministic for a given seed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

# Keep in sync with `benchmark.VERSION`. The benchmark validates this on the
# payload and raises if they drift, so a missed bump fails loudly rather than
# silently producing wrong numbers.
VERSION: int = 2

D_MODEL: int = 128
NOISE_SCALE: float = 0.1
SEED: int = 0
CANONICAL_COS: float = 0.7

# Number of independent noise draws each reported logit is averaged over. The
# payload still carries one float per quantity; averaging only reduces nuisance
# variation so two runs of the same model produce nearly the same metric.
N_NOISE_SAMPLES: int = 256

# Cosine-similarity sweep between each query vector and its target feature,
# i.e. ``c = cos(q_A, v_A) = cos(q_B, v_B)``. Must match `benchmark.COS_SWEEP`
# and the README "Payload contract" exactly (same values, same ascending order).
COSINE_SWEEP: list[float] = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]


@dataclass(frozen=True)
class Batch:
    """Fixed geometry + noise for a single cosine point.

    ``v_A, v_B`` are the orthogonal feature directions (shared across the whole
    sweep). ``q_A, q_B`` are this point's query vectors (cosine ``cosine`` with
    their own feature, orthogonal to the other feature and to each other).
    ``noise`` is the shared ``(N_NOISE_SAMPLES, d)`` block of isotropic noise,
    identical across cosine points so logit *differences* are not polluted by
    independent noise.
    """

    cosine: float
    d_model: int
    noise_scale: float
    q_A: np.ndarray
    q_B: np.ndarray
    v_A: np.ndarray
    v_B: np.ndarray
    noise: np.ndarray


# model_fn(q_A, q_B, x) -> (logit_A, logit_B).
ModelFn = Callable[[np.ndarray, np.ndarray, np.ndarray], "tuple[float, float]"]


def _orthonormal(rng: np.random.Generator, d: int, k: int) -> list[np.ndarray]:
    """Return ``k`` mutually orthonormal vectors in R^d, deterministic per rng."""
    m = rng.standard_normal((d, k))
    q, _ = np.linalg.qr(m)
    return [np.ascontiguousarray(q[:, i]) for i in range(k)]


def generate(seed: int = SEED) -> list[Batch]:
    """Return the full list of per-cosine batches, deterministically.

    Same seed -> byte-identical batches. The feature directions, the helper
    directions used to tilt the queries, and the noise block are all drawn once
    up front (independent of sweep length); only ``q_A, q_B`` change along the
    sweep. The geometry is the single source of truth for the question;
    attempts may not change it — they only provide ``model_fn``.
    """
    rng = np.random.default_rng(seed)
    # Four mutually orthonormal directions: the two feature directions and two
    # helper directions used to tilt the queries away from their features while
    # keeping q_A orthogonal to v_B (and q_B orthogonal to v_A, and q_A ⟂ q_B).
    v_A, v_B, u_A, u_B = _orthonormal(rng, D_MODEL, 4)
    noise = rng.standard_normal((N_NOISE_SAMPLES, D_MODEL))

    batches: list[Batch] = []
    for cosine in COSINE_SWEEP:
        c = float(cosine)
        s = math.sqrt(max(0.0, 1.0 - c * c))
        q_A = c * v_A + s * u_A  # unit; cos(q_A, v_A) == c, q_A ⟂ v_B
        q_B = c * v_B + s * u_B  # unit; cos(q_B, v_B) == c, q_B ⟂ v_A, q_A ⟂ q_B
        batches.append(
            Batch(
                cosine=c,
                d_model=D_MODEL,
                noise_scale=NOISE_SCALE,
                q_A=q_A,
                q_B=q_B,
                v_A=v_A,
                v_B=v_B,
                noise=noise,
            )
        )
    return batches


def _coerce_logits(out: Any, cosine: float, alpha: int, beta: int) -> tuple[float, float]:
    """Validate one ``model_fn`` return value: a pair of finite floats."""
    try:
        la, lb = out
    except (TypeError, ValueError):
        raise ValueError(
            f"model_fn(cos={cosine}, alpha={alpha}, beta={beta}) must return a "
            f"2-tuple (logit_A, logit_B), got {out!r}"
        )
    la_f = float(la)
    lb_f = float(lb)
    if not (math.isfinite(la_f) and math.isfinite(lb_f)):
        raise ValueError(
            f"model_fn(cos={cosine}, alpha={alpha}, beta={beta}) returned a "
            f"non-finite logit: ({la_f}, {lb_f})"
        )
    return la_f, lb_f


def evaluate(model_fn: ModelFn, *, seed: int = SEED) -> dict[str, Any]:
    """Run ``model_fn`` across the canonical cosine sweep, return a payload.

    For each cosine the model is probed on the four presence configurations,
    each averaged over :data:`N_NOISE_SAMPLES` noise draws::

        x(alpha, beta) = alpha * v_A + beta * v_B + noise_scale * eta

    The six reported per-slice logits match the README "Payload contract":

        logit_AA   = mean logit_A on x(1, 1)   (both present, A query)
        logit_AB   = mean logit_B on x(1, 1)   (both present, B query)
        logit_A0   = mean logit_A on x(1, 0)   (only A present)
        logit_B0   = mean logit_B on x(0, 1)   (only B present)
        logit_00_A = mean logit_A on x(0, 0)   (neither present)
        logit_00_B = mean logit_B on x(0, 0)   (neither present)

    Raises:
        ValueError: if ``model_fn`` returns the wrong shape or a non-finite value.
    """
    batches = generate(seed)
    eps = NOISE_SCALE
    n = N_NOISE_SAMPLES

    sweep: list[dict[str, Any]] = []
    for batch in batches:
        v_A, v_B = batch.v_A, batch.v_B
        q_A, q_B = batch.q_A, batch.q_B

        def mean_logits(alpha: int, beta: int) -> tuple[float, float]:
            la_sum = 0.0
            lb_sum = 0.0
            base = alpha * v_A + beta * v_B
            for i in range(n):
                x = base + eps * batch.noise[i]
                la, lb = _coerce_logits(
                    model_fn(q_A, q_B, x), batch.cosine, alpha, beta
                )
                la_sum += la
                lb_sum += lb
            return la_sum / n, lb_sum / n

        la_AA, lb_AA = mean_logits(1, 1)
        la_A0, _ = mean_logits(1, 0)
        _, lb_B0 = mean_logits(0, 1)
        la_00, lb_00 = mean_logits(0, 0)

        sweep.append(
            {
                "cos_sim": float(batch.cosine),
                "logit_AA": la_AA,
                "logit_AB": lb_AA,
                "logit_A0": la_A0,
                "logit_B0": lb_B0,
                "logit_00_A": la_00,
                "logit_00_B": lb_00,
            }
        )

    return {
        "version": VERSION,
        "d": D_MODEL,
        "noise_scale": NOISE_SCALE,
        "sweep": sweep,
        "canonical_cos_sim": CANONICAL_COS,
    }
