"""Minimal demo: attention as a soft AND via query superposition.

Setup:
  - d=2 with orthonormal concept directions e_A, e_B.
  - 7 hand-built keys with known (A-match, B-match) inner products with e_A, e_B.
  - Queries q_A = scale * e_A, q_B = scale * e_B, joint query q = q_A + q_B.
  - Score(token) = q . k_i = scale * (a_i + b_i).

We sweep over `scale` (a temperature-like knob) and dump:
  - softmax_weight: row-normalised exp(scores)
  - linear_weight:  baseline that uses the raw linear scores (shifted to be
                    non-negative, then normalised) — i.e. "what attention would
                    look like if there were no exp in the softmax".

The claim from the goal: under softmax, the "both" token's mass should sharply
dominate while the linear baseline stays diffuse. The artefact (weights.csv)
is what app.py renders.
"""

from __future__ import annotations

import csv
import json

import numpy as np
from agentic.experiments import results_dir

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


def main() -> None:
    out = results_dir(__file__)

    labels = [t[0] for t in TOKENS]
    a = np.array([t[1] for t in TOKENS], dtype=np.float64)
    b = np.array([t[2] for t in TOKENS], dtype=np.float64)
    raw_score = a + b  # q . k_i when q = q_A + q_B (unit-norm concept dirs)

    rows: list[dict[str, float | str]] = []
    for s in SCALES:
        scaled = raw_score * s
        exp_scores = np.exp(scaled)
        softmax_w = exp_scores / exp_scores.sum()

        shifted = scaled - scaled.min()
        total = shifted.sum()
        linear_w = shifted / total if total > 0 else np.full_like(shifted, 1.0 / len(shifted))

        for i, label in enumerate(labels):
            rows.append(
                {
                    "scale": s,
                    "token": label,
                    "a_match": float(a[i]),
                    "b_match": float(b[i]),
                    "linear_score": float(scaled[i]),
                    "softmax_weight": float(softmax_w[i]),
                    "linear_weight": float(linear_w[i]),
                }
            )

    csv_path = out / "weights.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    setup = {
        "tokens": [{"label": label, "a_match": am, "b_match": bm} for label, am, bm in TOKENS],
        "scales": SCALES,
        "note": (
            "q = q_A + q_B with |q_A| = |q_B| = scale; keys are 2D in the (e_A, e_B) "
            "basis so q . k_i = scale * (a_match + b_match)."
        ),
    }
    setup_path = out / "setup.json"
    setup_path.write_text(json.dumps(setup, indent=2))

    print(f"wrote {csv_path}")
    print(f"wrote {setup_path}")


if __name__ == "__main__":
    main()
