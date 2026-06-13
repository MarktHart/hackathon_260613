"""Task for the `attention_gcd` goal.

Synthetic GCD-decoding probe. Given integer pairs (a, b) presented to a small
transformer as the sequence [a, b, SEP], we ask whether gcd(a, b) is recovered
by the model's internals:

  * does any attention head's weight from SEP -> operands correlate with gcd?
  * is gcd linearly decodable from the residual stream at the SEP position?

Everything here is pure NumPy (no torch, no GPU). The attempt supplies a
`model_fn` that returns attention weights and residual activations; this module
reduces them into the payload that `benchmark.score` consumes.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

# A real model_fn maps tokens -> dict of activations.
ModelFn = Callable[[np.ndarray], Dict[str, Any]]

SEP_TOKEN = 200  # > MAX_N so it never collides with an operand value

# Per-slice bins over the true gcd value. Bigger-is-better accuracy is computed
# within each bin, so zero-variance bins (e.g. gcd == 1) are still well defined.
BINS = [
    ("g1", 1, 1),
    ("g2_3", 2, 3),
    ("g4_8", 4, 8),
    ("g9p", 9, 10 ** 9),
]


@dataclass(frozen=True)
class Batch:
    tokens: np.ndarray      # [batch, 3] int32: [a, b, SEP]
    gcd_labels: np.ndarray  # [batch] int32
    a_vals: np.ndarray      # [batch] int32
    b_vals: np.ndarray      # [batch] int32
    max_n: int
    batch_size: int


def generate(seed: int = 0, max_n: int = 100, batch_size: int = 512) -> Batch:
    """Deterministic for a given seed: same seed -> identical batch."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, max_n + 1, size=batch_size, dtype=np.int32)
    b = rng.integers(1, max_n + 1, size=batch_size, dtype=np.int32)
    gcd = np.gcd(a.astype(np.int64), b.astype(np.int64)).astype(np.int32)
    sep = np.full(batch_size, SEP_TOKEN, dtype=np.int32)
    tokens = np.stack([a, b, sep], axis=1).astype(np.int32)
    return Batch(
        tokens=tokens,
        gcd_labels=gcd,
        a_vals=a,
        b_vals=b,
        max_n=max_n,
        batch_size=batch_size,
    )


# --------------------------------------------------------------------------- #
# Small numeric helpers (pure NumPy)
# --------------------------------------------------------------------------- #
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return 0.0
    sx = x.std()
    sy = y.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    r = np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy)
    return float(np.clip(r, -1.0, 1.0))


def _ridge_probe(X: np.ndarray, y: np.ndarray, n_train: int, lam: float = 1.0):
    """Fit a ridge probe on the first `n_train` rows, evaluate on the rest.

    Returns (test_r2, test_predictions, test_targets). Features are
    standardised with train statistics; a bias column is appended.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:], y[n_train:]
    if Xtr.shape[0] < 2 or Xte.shape[0] < 1:
        return 0.0, np.zeros(Xte.shape[0]), yte

    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], axis=1)
    Xte = np.concatenate([Xte, np.ones((Xte.shape[0], 1))], axis=1)

    D = Xtr.shape[1]
    A = Xtr.T @ Xtr + lam * np.eye(D)
    try:
        w = np.linalg.solve(A, Xtr.T @ ytr)
    except np.linalg.LinAlgError:
        return 0.0, np.zeros(Xte.shape[0]), yte
    pred = Xte @ w

    ss_res = float(np.sum((yte - pred) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(r2), pred, yte


def _acc(pred: np.ndarray, target: np.ndarray) -> float:
    if target.size == 0:
        return 0.0
    return float(np.mean(np.rint(pred) == target))


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    model_fn: ModelFn,
    batch: Batch | None = None,
    max_n: int = 100,
    batch_size: int = 512,
) -> dict:
    """Run `model_fn` over a batch and return the benchmark payload."""
    if batch is None:
        batch = generate(seed=42, max_n=max_n, batch_size=batch_size)

    out = model_fn(batch.tokens)
    if not isinstance(out, dict):
        raise ValueError("model_fn must return a dict")
    attn_list: List[np.ndarray] = list(out["attn_weights"])
    resid_list: List[np.ndarray] = list(out["resid_post"])
    if len(attn_list) == 0 or len(resid_list) == 0:
        raise ValueError("model_fn returned empty attn_weights / resid_post")

    n_layers = len(resid_list)
    n_heads = int(attn_list[0].shape[1])
    d_model = int(resid_list[0].shape[2])
    seq_len = int(batch.tokens.shape[1])
    sep_idx = seq_len - 1

    n = batch.batch_size
    n_train = n // 2
    n_test = n - n_train
    gcd = batch.gcd_labels.astype(np.float64)

    # --- Attention: per (layer, head) correlation of SEP->operands weight w/ gcd
    attn_corr: List[List[float]] = []
    for L in range(len(attn_list)):
        row: List[float] = []
        a = np.asarray(attn_list[L], dtype=np.float64)
        for H in range(a.shape[1]):
            sep_to_ops = a[:, H, sep_idx, :2].mean(axis=1)  # [batch]
            row.append(_pearson(sep_to_ops, gcd))
        attn_corr.append(row)

    # Baseline "attention" reference: correlation of the raw value feature
    # (a + b) with gcd. A learned head should beat this trivial correlate.
    baseline_attn_corr = _pearson(
        (batch.a_vals + batch.b_vals).astype(np.float64), gcd
    )

    # --- Residual probe: gcd decodability from resid_post at SEP, per layer
    resid_r2: List[float] = []
    test_preds: List[np.ndarray] = []
    for L in range(n_layers):
        X = np.asarray(resid_list[L], dtype=np.float64)[:, sep_idx, :]  # [batch, D]
        r2, pred, _ = _ridge_probe(X, gcd, n_train)
        resid_r2.append(r2)
        test_preds.append(pred)

    gcd_test = gcd[n_train:]

    # Baseline probe: gcd from the raw operand values [a, b] (linear, no model).
    raw = np.stack(
        [batch.a_vals.astype(np.float64), batch.b_vals.astype(np.float64)], axis=1
    )
    base_r2, base_pred, _ = _ridge_probe(raw, gcd, n_train)

    # Global test accuracies (best-layer reduction happens in score()).
    resid_acc_global = [_acc(p, gcd_test) for p in test_preds]
    base_acc_global = _acc(base_pred, gcd_test)

    # --- Per-slice sweep over gcd bins (on the held-out test split)
    sweep: List[Dict[str, Any]] = []
    for name, lo, hi in BINS:
        mask = (gcd_test >= lo) & (gcd_test <= hi)
        count = int(mask.sum())
        if count > 0:
            resid_acc = [_acc(p[mask], gcd_test[mask]) for p in test_preds]
            base_acc = _acc(base_pred[mask], gcd_test[mask])
        else:
            resid_acc = [0.0 for _ in test_preds]
            base_acc = 0.0
        sweep.append(
            {
                "bin": name,
                "lo": int(lo),
                "hi": int(hi),
                "count": count,
                "resid_acc": resid_acc,
                "baseline_acc": base_acc,
            }
        )

    return {
        "version": 1,
        "config": {
            "max_n": batch.max_n,
            "batch_size": batch.batch_size,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_model": d_model,
            "n_train": n_train,
            "n_test": n_test,
            "sep_idx": sep_idx,
        },
        "attn_corr": attn_corr,
        "baseline_attn_corr": float(baseline_attn_corr),
        "global": {
            "resid_r2": [float(v) for v in resid_r2],
            "resid_acc": [float(v) for v in resid_acc_global],
            "baseline_r2": float(base_r2),
            "baseline_acc": float(base_acc_global),
        },
        "sweep": sweep,
    }


def random_model_fn() -> ModelFn:
    """A null model_fn with the real signature: uniform attention + random
    residuals. Pure NumPy. Produces a degenerate (≈baseline) payload."""

    def _fn(tokens: np.ndarray) -> Dict[str, Any]:
        tokens = np.asarray(tokens)
        batch, seq_len = tokens.shape
        n_layers, n_heads, d_model = 2, 4, 128
        rng = np.random.default_rng(0)
        # Uniform attention carries no information about gcd.
        attn = np.full((batch, n_heads, seq_len, seq_len), 1.0 / seq_len, dtype=np.float32)
        resid = rng.normal(0.0, 1.0, size=(batch, seq_len, d_model)).astype(np.float32)
        return {
            "attn_weights": [attn for _ in range(n_layers)],
            "resid_post": [resid.copy() for _ in range(n_layers)],
        }

    return _fn


if __name__ == "__main__":
    b = generate(seed=123)
    payload = evaluate(random_model_fn(), b)
    print("keys:", list(payload.keys()))
    print("sweep bins:", [r["bin"] for r in payload["sweep"]])
    print("global:", payload["global"])
