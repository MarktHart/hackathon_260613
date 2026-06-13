"""Per-experiment conventions.

Layout: `experiments/<goal>/<attempt_name>/` is one attempt at one goal. The
goal directory owns a `benchmark.py` defining the metric every attempt is
judged against; attempts hand it data via `record_benchmark` and display
history via `benchmark_panel` inside their Gradio app.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def results_dir(caller_file: str | Path, run_id: str | None = None) -> Path:
    """Create and return a results directory for the calling experiment.

    The directory lives at `<attempt>/results/<run_id>/`. `run_id` defaults
    to a UTC timestamp so concurrent runs don't collide.
    """
    base = Path(caller_file).resolve().parent / "results"
    run = run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = base / run
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_task(caller_file: str | Path) -> Any:
    """Import the enclosing goal's `task.py` and return the module.

    `task.py` lives at the goal level (sibling of `benchmark.py`) and owns the
    synthetic data generator + canonical evaluator. Every attempt at the same
    goal imports it via this helper, so the data is byte-identical across
    attempts. Returns the imported module — typically used as
    `task = load_task(__file__); task.evaluate(my_model_fn)`.
    """
    attempt_dir = Path(caller_file).resolve().parent
    goal_dir = attempt_dir.parent
    task_path = goal_dir / "task.py"
    if not task_path.is_file():
        raise FileNotFoundError(
            f"Expected {task_path} to exist — the goal owns the task definition."
        )
    spec = importlib.util.spec_from_file_location(f"_goal_task_{goal_dir.name}", task_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {task_path}.")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses (and similar) can look the module up.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def record_benchmark(
    caller_file: str | Path,
    run_dir: Path,
    payload: dict[str, Any],
) -> Path:
    """Score this run using the enclosing goal's benchmark.py and dump benchmark.json.

    `caller_file` is the attempt's `__file__`. We walk up two levels to find
    `<goal>/benchmark.py`, import it dynamically, call its `score(payload)`
    function, and write `{goal, attempt, run_id, metrics, payload}` to
    `<run_dir>/benchmark.json`. The goal owns the metric so every attempt is
    judged on the same yardstick — attempts only ever hand over data, never
    define what counts.
    """
    attempt_dir = Path(caller_file).resolve().parent
    goal_dir = attempt_dir.parent
    bench_path = goal_dir / "benchmark.py"
    if not bench_path.is_file():
        raise FileNotFoundError(f"Expected {bench_path} to exist — the goal owns the benchmark.")

    spec = importlib.util.spec_from_file_location(f"_goal_benchmark_{goal_dir.name}", bench_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {bench_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    score_fn = getattr(module, "score", None)
    if score_fn is None:
        raise AttributeError(f"{bench_path} must export `score(payload) -> dict[str, float]`.")
    metrics: dict[str, Any] = score_fn(payload)

    record = {
        "goal": goal_dir.name,
        "attempt": attempt_dir.name,
        "run_id": run_dir.name,
        "metrics": metrics,
        "payload": payload,
    }
    out = run_dir / "benchmark.json"
    out.write_text(json.dumps(record, indent=2, default=str))
    return out


def benchmark_panel(goal_dir: str | Path) -> None:
    """Render the cross-attempt benchmark history for a goal as Gradio components.

    Call from inside a `gr.Blocks` / `gr.Tab` context. Scans
    `<goal_dir>/*/results/*/benchmark.json` across every attempt and renders:

    - a leaderboard of the latest value per (attempt × metric);
    - a line chart of a chosen metric over chronologically-sorted runs,
      coloured by attempt.

    Heavy imports (gradio, matplotlib) are deferred so importing
    `agentic.experiments` from a non-UI context stays cheap.
    """
    import gradio as gr
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    goal = Path(goal_dir).resolve()

    def _load_rows_all() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for bench_file in sorted(goal.glob("*/results/*/benchmark.json")):
            try:
                data = json.loads(bench_file.read_text())
            except json.JSONDecodeError:
                continue
            attempt = data.get("attempt", bench_file.parents[2].name)
            run_id = data.get("run_id", bench_file.parent.name)
            metrics = data.get("metrics") or {}
            version = metrics.get("version")
            for name, value in metrics.items():
                if name == "version":
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                rows.append(
                    {
                        "attempt": attempt,
                        "run_id": run_id,
                        "version": version,
                        "metric": name,
                        "value": float(value),
                    }
                )
        return rows

    def _latest_version(rows: list[dict[str, Any]]) -> int | None:
        versions = {r["version"] for r in rows if isinstance(r["version"], int)}
        return max(versions) if versions else None

    def _load_rows() -> list[dict[str, Any]]:
        """Rows filtered to the latest VERSION present — older runs stay on disk
        but don't get mixed into the active series."""
        rows = _load_rows_all()
        latest = _latest_version(rows)
        if latest is None:
            return rows
        return [r for r in rows if r["version"] == latest]

    def _plot(metric: str) -> Figure:
        sub = [r for r in _load_rows() if r["metric"] == metric]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        if not sub:
            ax.text(0.5, 0.5, "no benchmark data yet", ha="center", va="center")
            ax.set_axis_off()
            return fig
        by_attempt: dict[str, list[tuple[str, float]]] = {}
        for r in sub:
            by_attempt.setdefault(r["attempt"], []).append((r["run_id"], r["value"]))
        for attempt, pairs in sorted(by_attempt.items()):
            pairs.sort()
            xs = list(range(len(pairs)))
            ys = [v for _, v in pairs]
            ax.plot(xs, ys, marker="o", label=attempt)
        ax.set_xlabel("run (chronological)")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} — across attempts")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _leaderboard() -> list[list[Any]]:
        latest: dict[tuple[str, str], tuple[str, float]] = {}
        for r in _load_rows():
            key = (r["attempt"], r["metric"])
            prev = latest.get(key)
            if prev is None or r["run_id"] > prev[0]:
                latest[key] = (r["run_id"], r["value"])
        attempts = sorted({a for (a, _) in latest})
        metrics = sorted({m for (_, m) in latest})
        table: list[list[Any]] = []
        for a in attempts:
            row: list[Any] = [a]
            for m in metrics:
                v = latest.get((a, m))
                row.append(round(v[1], 4) if v else None)
            table.append(row)
        return table

    all_rows = _load_rows_all()
    rows = _load_rows()
    all_metrics = sorted({r["metric"] for r in rows})
    latest = _latest_version(all_rows)
    older_runs = len({(r["attempt"], r["run_id"]) for r in all_rows if r["version"] != latest})

    header = f"## Benchmark history — `{goal.name}`"
    if latest is not None:
        header += f" (v{latest})"
    gr.Markdown(header)
    if older_runs:
        gr.Markdown(f"_{older_runs} older-version run(s) hidden._")
    if not rows:
        gr.Markdown("_No `benchmark.json` files yet. Run an attempt's `main.py` to populate._")
        return

    initial_metric = all_metrics[0]
    leaderboard_headers = ["attempt", *sorted({r["metric"] for r in rows})]

    with gr.Row():
        metric_dd = gr.Dropdown(choices=all_metrics, value=initial_metric, label="Metric")
        refresh = gr.Button("Refresh", size="sm")
    leaderboard = gr.DataFrame(
        value=_leaderboard(),
        headers=leaderboard_headers,
        label="Latest per attempt",
        interactive=False,
    )
    plot = gr.Plot(value=_plot(initial_metric))

    metric_dd.change(_plot, inputs=metric_dd, outputs=plot)
    refresh.click(
        lambda m: (_plot(m), _leaderboard()),
        inputs=metric_dd,
        outputs=[plot, leaderboard],
    )
