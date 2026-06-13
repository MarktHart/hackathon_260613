"""Schema for jury verdicts.

The jury reads an attempt + its goal and writes one `verdict.json` next to
the attempt's `benchmark.json`. Each human-judged rubric criterion gets a
score in [1, 5] and a one-line justification.

The file format is the contract — if you change keys here, bump VERDICT_VERSION
so downstream readers (dashboard, cross-attempt review) can switch on it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

VERDICT_VERSION = 1

# Overall grade as a pure function of the mean rubric score (1–5). This is the
# single source of truth for the verdict label — the jury scores each criterion,
# and the grade falls out of the average so it can't drift from the numbers.
GRADES = ("fail", "borderline", "good", "perfect", "unscored")


def grade_from_score(score: float | None) -> str:
    """Map a mean rubric score in [1, 5] to an overall grade.

    Thresholds: <2 → ``fail``, <4 → ``borderline``, <5 → ``good``, ==5 →
    ``perfect``. ``None`` (no scored criteria) → ``unscored``.
    """
    if score is None:
        return "unscored"
    if score < 2:
        return "fail"
    if score < 4:
        return "borderline"
    if score < 5:
        return "good"
    return "perfect"


@dataclass
class CriterionScore:
    score: int  # 1 (broken) to 5 (excellent); 0 = not applicable
    note: str  # one-line justification


@dataclass
class Verdict:
    """A complete grade for one attempt at one goal."""

    goal: str
    attempt: str
    run_id: str | None = None
    verdict_version: int = VERDICT_VERSION

    architecture_fit: CriterionScore | None = None
    baseline_comparison: CriterionScore | None = None
    faithfulness: CriterionScore | None = None
    operating_range: CriterionScore | None = None
    hardcoded_weights_bonus: CriterionScore | None = None
    visual_judgement: CriterionScore | None = None
    visualisation_rationale: CriterionScore | None = None

    automated_metrics: dict[str, Any] = field(default_factory=dict)

    overall: Literal["perfect", "good", "borderline", "fail", "unscored"] = "unscored"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def write(self, path: Path) -> Path:
        path.write_text(self.to_json())
        return path

    @classmethod
    def load(cls, path: Path) -> Verdict:
        data = json.loads(Path(path).read_text())
        crit_keys = [
            "architecture_fit",
            "baseline_comparison",
            "faithfulness",
            "operating_range",
            "hardcoded_weights_bonus",
            "visual_judgement",
            "visualisation_rationale",
        ]
        for k in crit_keys:
            if isinstance(data.get(k), dict):
                data[k] = CriterionScore(**data[k])
        return cls(**data)


# Embedded in JURY prompt so the model returns matching JSON without reading source.
JURY_OUTPUT_SCHEMA = """\
{
  "goal": "<goal slug>",
  "attempt": "<attempt name>",
  "run_id": "<latest run id under results/>",
  "verdict_version": 1,
  "architecture_fit":         {"score": 1-5, "note": "one line"},
  "baseline_comparison":      {"score": 1-5, "note": "one line"},
  "faithfulness":             {"score": 1-5, "note": "one line"},
  "operating_range":          {"score": 1-5, "note": "one line"},
  "hardcoded_weights_bonus":  {"score": 0-5, "note": "one line; 0 if N/A"},
  "visual_judgement":         {"score": 1-5, "note": "one line"},
  "visualisation_rationale":  {"score": 1-5, "note": "one line"},
  "automated_metrics":        {"<copied from benchmark.json>": ...},
  "overall":                  "perfect" | "good" | "borderline" | "fail",
  "notes":                    "two-three sentence summary"
}
"""
