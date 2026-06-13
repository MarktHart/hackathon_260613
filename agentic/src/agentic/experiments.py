"""Per-experiment conventions.

Every experiment is a uv workspace member under `experiments/<category>/<name>/`
and writes its outputs to `results/<run-id>/` inside its own folder. Use
`results_dir(__file__)` from an experiment script to get a fresh, timestamped
directory you can write artefacts into.
"""

from datetime import datetime
from pathlib import Path


def results_dir(caller_file: str | Path, run_id: str | None = None) -> Path:
    """Create and return a results directory for the calling experiment.

    The directory lives at `<experiment>/results/<run_id>/`. `run_id` defaults
    to a UTC timestamp so concurrent runs don't collide.
    """
    base = Path(caller_file).resolve().parent / "results"
    run = run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = base / run
    out.mkdir(parents=True, exist_ok=True)
    return out
