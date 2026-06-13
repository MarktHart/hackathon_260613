"""Minimal demo: attention as a soft AND via query superposition.

Setup
-----
2D concept basis (e_1, e_2). The two concept directions are
    q_A = e_1
    q_B = cos * e_1 + sqrt(1 - cos^2) * e_2          (cos = q_A . q_B)
Both are unit-norm before scaling. The joint query is q = q_A + q_B, so
    q . k_i = scale * ((1 + cos) * a_match + sqrt(1 - cos^2) * b_match)
for a key k_i = a_match * e_1 + b_match * e_2.

We hand-build 7 tokens with fixed (a_match, b_match) so the dot products are
interpretable: two A-only (3, -1), two B-only (-1, 3), one `both` (3, 3), two
`neither` (-1, -1).

Two sweeps
----------
- Demo (`weights.csv`): scale ∈ [0.25, 3.0] at cos = 0 (orthogonal queries).
  Drives the per-token bar chart in the Gradio Demo tab.
- Benchmark: cos ∈ {0.0, 0.3, 0.5, 0.7, 0.9} at the canonical scale = 1.0.
  Drives the goal's `score()` — the superposition-robustness metric.
"""

from __future__ import annotations

import csv
import json
import math

import numpy as np
from agentic.experiments import record_benchmark, results_dir

TOKENS: list[tuple[str, float, float]] = [
    ("A_only_1", 3.0, -1.0),
    ("A_only_2", 3.0, -1.0),
    ("B_only_1", -1.0, 3.0),
    ("B_only_2", -1.0, 3.0),
    ("both", 3.0, 3.0),
    ("neither_1", -1.0, -1.0),
    ("neither_2", -1.0, -1.0),
]

SCALES: list[float] = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
COSINES: list[float] = [0.0, 0.3, 0.5, 0.7, 0.9]
CANONICAL_SCALE: float = 1.0


def _scores(scale: float, cosine: float) -> np.ndarray:
    """Per-token q . k under the parameterisation in the module docstring."""
    sqrt_term = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    return np.array(
        [scale * ((1.0 + cosine) * a + sqrt_term * b) for _, a, b in TOKENS],
        dtype=np.float64,
    )


def _attention(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (softmax_weights, linear_baseline_weights) for one set of scores."""
    exp_scores = np.exp(scores)
    softmax_w = exp_scores / exp_scores.sum()
    shifted = scores - scores.min()
    total = shifted.sum()
    linear_w = shifted / total if total > 0 else np.full_like(shifted, 1.0 / len(shifted))
    return softmax_w, linear_w


def main() -> None:
    out = results_dir(__file__)
    labels = [t[0] for t in TOKENS]

    rows: list[dict[str, float | str]] = []
    for s in SCALES:
        softmax_w, linear_w = _attention(_scores(scale=s, cosine=0.0))
        for i, label in enumerate(labels):
            rows.append(
                {
                    "scale": s,
                    "token": label,
                    "softmax_weight": float(softmax_w[i]),
                    "linear_weight": float(linear_w[i]),
                }
            )
    csv_path = out / "weights.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sweep: list[dict[str, object]] = []
    for cos in COSINES:
        softmax_w, linear_w = _attention(_scores(scale=CANONICAL_SCALE, cosine=cos))
        sweep.append(
            {
                "cosine": cos,
                "softmax_weights": {label: float(softmax_w[i]) for i, label in enumerate(labels)},
                "linear_weights": {label: float(linear_w[i]) for i, label in enumerate(labels)},
            }
        )

    payload = {
        "sweep": sweep,
        "both_label": "both",
        "single_feature_labels": ["A_only_1", "A_only_2", "B_only_1", "B_only_2"],
        "canonical_scale": CANONICAL_SCALE,
    }
    bench_path = record_benchmark(__file__, out, payload)

    setup = {
        "tokens": [{"label": label, "a_match": am, "b_match": bm} for label, am, bm in TOKENS],
        "scales": SCALES,
        "cosines": COSINES,
        "canonical_scale": CANONICAL_SCALE,
        "note": (
            "q_A = e_1, q_B = cos*e_1 + sqrt(1-cos^2)*e_2 (unit-norm before scaling). "
            "Keys k_i = a_match*e_1 + b_match*e_2. Score = scale*((1+cos)*a + sqrt(1-cos^2)*b)."
        ),
    }
    setup_path = out / "setup.json"
    setup_path.write_text(json.dumps(setup, indent=2))

    print(f"wrote {csv_path}")
    print(f"wrote {setup_path}")
    print(f"wrote {bench_path}")


if __name__ == "__main__":
    main()
