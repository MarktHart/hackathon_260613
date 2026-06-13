"""Task definition for the `attention_xor` goal.

The goal owns this file just like it owns ``benchmark.py``. Every attempt
imports it so the token vocabulary, the marginal sweep, the canonical ``p``,
the evaluation seed and the linear baseline are byte-identical across attempts
— no drift in what the question actually is.

An attempt contributes **only** a model function with the signature documented
in ``README.md``::

    def model_fn(tokens: np.ndarray) -> np.ndarray:
        # tokens : (N, 4) int array, vocabulary in the README
        # returns: (N,) float array of logits; XOR=1 iff logit > 0

``evaluate`` runs that callable across the full marginal sweep and assembles
the payload dict that ``benchmark.score`` consumes — pass it straight to
``record_benchmark``.

Pure Python + NumPy; no I/O, no network. Deterministic for a given seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# Keep in sync with `benchmark.VERSION`. The benchmark validates this on the
# payload and raises if they drift, so a missed bump fails loudly.
VERSION: int = 1

# Token vocabulary (see README): 0=CLS, 1=A0, 2=A1, 3=B0, 4=B1, 5=SEP.
CLS, SEP = 0, 5

# Marginal sweep p = P(A=1) = P(B=1). Must match `benchmark.SWEEP_PS` exactly
# (same values, same ascending order).
SWEEP_PS: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CANONICAL_P: float = 0.5

N_PER_SLICE: int = 1000
EVAL_SEED: int = 42

# model_fn(tokens: (N,4) int) -> (N,) float logits; predict XOR=1 iff logit > 0.
ModelFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Batch:
    """One sweep point: the tokens, the labels, and the marginal that made them."""

    tokens: np.ndarray  # (N, 4) int32
    labels: np.ndarray  # (N,)  int32 in {0, 1}  (A XOR B)
    p: float


def generate(seed: int = EVAL_SEED) -> list[Batch]:
    """Return one :class:`Batch` per marginal in :data:`SWEEP_PS`, deterministically.

    Same seed -> byte-identical batches. ``A`` and ``B`` are independent
    Bernoulli(``p``) draws; the label is ``A XOR B``. Tokens are encoded as
    ``[CLS, A_tok, B_tok, SEP]`` with ``A_tok in {1,2}`` for ``A in {0,1}`` and
    ``B_tok in {3,4}`` for ``B in {0,1}``.
    """
    rng = np.random.default_rng(seed)
    batches: list[Batch] = []
    for p in SWEEP_PS:
        n = N_PER_SLICE
        A = rng.binomial(1, p, size=n).astype(np.int32)
        B = rng.binomial(1, p, size=n).astype(np.int32)
        labels = (A ^ B).astype(np.int32)
        tokens = np.zeros((n, 4), dtype=np.int32)
        tokens[:, 0] = CLS
        tokens[:, 1] = A + 1  # 1 (A=0) or 2 (A=1)
        tokens[:, 2] = B + 3  # 3 (B=0) or 4 (B=1)
        tokens[:, 3] = SEP
        batches.append(Batch(tokens=tokens, labels=labels, p=float(p)))
    return batches


def _linear_baseline_accuracy(tokens: np.ndarray, labels: np.ndarray) -> float:
    """Best in-sample accuracy of any linear probe over the one-hot ``(A, B)``.

    XOR is not linearly separable in ``[A=0, A=1, B=0, B=1, bias]`` — no linear
    threshold classifies all four input cells correctly — but a linear probe can
    still beat the majority class by giving up on exactly one cell (e.g. predict
    ``A OR B``, or isolate a single corner). The honest linear floor is therefore
    the *best* linear probe, not the constant predictor.

    A linear threshold over the one-hot features can realise any boolean function
    of ``(A, B)`` except the two non-separable ones, XOR and XNOR. So we
    enumerate all 16 boolean predictors, drop those two, and take the maximum
    empirical accuracy on this slice's data. This is always ``>= max(P(XOR=1),
    P(XOR=0))`` (the all-0 / all-1 predictors are included) and in expectation
    equals ``1 - min(p, 1-p)**2`` (0.75 at p=0.5, ~0.99 at p=0.1). A genuine XOR
    mechanism must beat this stronger floor.
    """
    A = (np.asarray(tokens)[:, 1] - 1).astype(np.int64)  # 1/2 -> 0/1
    B = (np.asarray(tokens)[:, 2] - 3).astype(np.int64)  # 3/4 -> 0/1
    y = np.asarray(labels).astype(np.int64)
    cell = 2 * A + B  # cell index: 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1)

    best = 0.0
    for bits in range(16):
        lut = np.array([(bits >> k) & 1 for k in range(4)], dtype=np.int64)
        # Skip the only non-linearly-separable dichotomies: XOR and XNOR.
        t = (int(lut[0]), int(lut[1]), int(lut[2]), int(lut[3]))
        if t == (0, 1, 1, 0) or t == (1, 0, 0, 1):
            continue
        acc = float(np.mean(lut[cell] == y))
        if acc > best:
            best = acc
    return best


def _coerce_logits(out: Any, p: float, n: int) -> np.ndarray:
    """Validate one ``model_fn`` return value: a finite real vector of length n."""
    arr = np.asarray(out, dtype=np.float64).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(
            f"model_fn(p={p}) must return a length-{n} logit vector, "
            f"got shape {np.asarray(out).shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"model_fn(p={p}) returned non-finite logits")
    return arr


def evaluate(model_fn: ModelFn, *, seed: int = EVAL_SEED) -> dict[str, Any]:
    """Run ``model_fn`` across the marginal sweep and return the benchmark payload.

    For each slice the model is given the ``(N, 4)`` token batch and must return
    an ``(N,)`` logit vector; the prediction is ``XOR=1`` iff ``logit > 0``.

    Raises:
        ValueError: if ``model_fn`` returns the wrong shape or a non-finite value.
    """
    batches = generate(seed)
    sweep: list[dict[str, Any]] = []
    for batch in batches:
        n = int(batch.tokens.shape[0])
        logits = _coerce_logits(model_fn(batch.tokens), batch.p, n)
        preds = (logits > 0.0).astype(np.int32)
        acc = float(np.mean(preds == batch.labels))
        baseline = _linear_baseline_accuracy(batch.tokens, batch.labels)
        sweep.append(
            {
                "p": float(batch.p),
                "accuracy": acc,
                "baseline_accuracy": baseline,
                "frac_xor1": float(np.mean(batch.labels)),
                "n": n,
            }
        )

    return {
        "version": VERSION,
        "canonical_p": CANONICAL_P,
        "n_per_slice": N_PER_SLICE,
        "sweep": sweep,
    }


def random_model_fn() -> ModelFn:
    """A contract-compatible model that returns all-zero logits (predicts XOR=0).

    Pure NumPy, no torch, no GPU. Used by the pipeline smoke test:
    ``benchmark.score(evaluate(random_model_fn()))`` must run cleanly.
    """

    def _fn(tokens: np.ndarray) -> np.ndarray:
        n = int(np.asarray(tokens).shape[0])
        return np.zeros((n,), dtype=np.float64)

    return _fn
