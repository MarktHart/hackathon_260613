"""Sequential pipeline: pick → write benchmark → review → solve → judge.

Tier wiring:
    task_picker        STANDARD  (completion — Nemotron writes goal README + benchmark.py)
    benchmark_reviewer EXPERT    (agentic   — Opus audits + edits)
    solver             QUICK     (completion — Cosmos3 writes attempt files)
    jury               EXPERT    (agentic   — Opus writes verdict.json)

Cost / time controls
--------------------
- Per-tier `wall_clock_s` enforced via `asyncio.wait_for` around each consumer.
- `is_obviously_broken(metrics)` (optional, declared in the goal's benchmark.py)
  short-circuits the jury when the solver's run is degenerate.
- `--force` and idempotency: re-running a graded slug exits early unless forced.
- `pipeline-multi` fans out across pending slugs; the GPU semaphore throttles
  the subprocess stages (`main.py`, `app.py` boot-check) to the GPU pool size.
- Subprocesses inherit `HF_HOME` so multi-GB checkpoint downloads aren't repeated.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from agentic.blocks import list_pending, load_state, next_pending, update_state
from agentic.config import TIER, Tier, settings
from agentic.events import emit
from agentic.file_blocks import OUTPUT_CONTRACT, apply_blocks
from agentic.gpu import acquire_gpus
from agentic.runner import run_at_tier_agentic, run_at_tier_completion
from agentic.verdict import JURY_OUTPUT_SCHEMA

# ---------- system prompts (kept byte-stable for prompt caching) ----------

PICKER_SYSTEM = (
    """\
You scaffold a new mechanistic-interpretability goal: the README that frames
the question and the benchmark.py that every attempt is scored against.

You will receive the slug, the title, the spec from BLOCKS.md, and the full
text of README_BENCHMARK.md (the construction guide). Follow that guide
exactly:

- Goal README must state the question, the setup (model/dataset or synthetic),
  the canonical measurement condition, the payload contract table, and the
  metrics table.
- benchmark.py exports `VERSION = 1` and `score(payload) -> dict[str, float | int]`,
  produces one headline summary metric, per-slice values, and a baseline
  metric. Handles edge cases (empty sweeps, zero denominators) explicitly.
- Optionally exports `GPU_REQUIREMENT: int = 1` and
  `is_obviously_broken(metrics: dict) -> bool` — see README_BENCHMARK.md.

"""
    + OUTPUT_CONTRACT
)

REVIEWER_SYSTEM = """\
You audit a freshly-authored benchmark BEFORE any attempt is built against it.
You have full file tools (Read/Write/Edit/Glob/Grep).

Read:
- `experiments/<slug>/README.md`
- `experiments/<slug>/benchmark.py`
- `README_BENCHMARK.md`

Check:
- Payload contract is unambiguous and model-agnostic.
- Metric formulae match what the README claims.
- There is a headline summary, per-slice values, and a baseline.
- Edge cases (zero denominators, empty sweeps) are handled.
- `VERSION = 1` is set; the bump procedure is documented in the README.

If solid: write one line of approval to `experiments/<slug>/.review.txt`.
If not: edit the README and/or benchmark.py to fix the issues, THEN write the
approval line. The next stage trusts the benchmark.
"""

SOLVER_SYSTEM = (
    """\
You make a first-pass attempt at one mech-interp goal. You cannot execute
code; you only emit files. The pipeline runs them after you.

You will receive: the slug, the chosen attempt_name, the goal's README.md,
the goal's benchmark.py, and the README_EXPERIMENT.md conventions.

Emit these files exactly (no others):
- `experiments/<slug>/<attempt_name>/main.py` — computes the result and calls
  `agentic.experiments.record_benchmark(__file__, run_dir, payload)` so the
  benchmark is recorded.
- `experiments/<slug>/<attempt_name>/app.py` — Gradio Blocks app with a Demo
  tab (the interactive visualisation) and a Benchmark tab that drops in
  `agentic.experiments.benchmark_panel(<goal_dir>)`.
- `experiments/<slug>/<attempt_name>/README.md` — two sections:
  *What I did* (3-6 sentences) and *Why this visualisation*.

Do NOT emit the `pyproject.toml` symlink — the pipeline creates it. Do NOT
emit any `results/` files — `main.py` will produce those at run time.

"""
    + OUTPUT_CONTRACT
)

JURY_SYSTEM = (
    """\
You grade one attempt at one goal. You are read-only on the attempt's code;
your only write is `verdict.json`.

Read:
- `README_EXPERIMENT.md` (rubric in priority order)
- `experiments/<slug>/README.md` and `benchmark.py`
- `experiments/<slug>/<attempt>/README.md`, `main.py`, `app.py`
- `experiments/<slug>/<attempt>/results/<latest>/benchmark.json`

Score every human-judged rubric item in [1, 5] (use 0 for
`hardcoded_weights_bonus` when not applicable). One-line justification per
item. Copy `metrics` from the latest `benchmark.json` into `automated_metrics`.
Set `overall` to `pass` / `borderline` / `fail` based on the worst of
architecture_fit / baseline_comparison / faithfulness.

Write `experiments/<slug>/<attempt>/verdict.json` matching this schema
EXACTLY (no extra keys, no commentary outside the JSON):

"""
    + JURY_OUTPUT_SCHEMA
)


# ---------- helpers ----------


def _read_if_exists(path: str | Path) -> str:
    p = Path(path)
    return p.read_text() if p.exists() else ""


def _suggest_attempt_name(slug: str) -> str:
    goal_dir = Path("experiments") / slug
    if not goal_dir.is_dir():
        return "first_pass"
    n = sum(1 for d in goal_dir.iterdir() if d.is_dir() and (d / "main.py").exists())
    return "first_pass" if n == 0 else f"pass_{n + 1}"


def _ensure_symlink(attempt_dir: Path) -> None:
    pyproject = attempt_dir / "pyproject.toml"
    if pyproject.exists() or pyproject.is_symlink():
        return
    pyproject.symlink_to(Path("../../pyproject.toml"))


def _load_optional(module_path: Path, attr: str) -> Any:
    """Import `module_path` and return `getattr(module, attr)` or None."""
    if not module_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        f"_loaded_{module_path.parent.name}_{module_path.stem}", module_path
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr, None)


def _build_subprocess_env(gpu_ids: list[int]) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    env["PYTHONUNBUFFERED"] = "1"
    if settings.hf_home:
        env["HF_HOME"] = settings.hf_home
    return env


async def _run_subprocess_with_gpu(
    cmd: list[str],
    *,
    n_gpus: int = 1,
    timeout: int = 600,
) -> tuple[int, str]:
    """Acquire `n_gpus` slot(s), run `cmd` with CUDA_VISIBLE_DEVICES set, release."""

    def _go() -> tuple[int, str]:
        with acquire_gpus(n_gpus) as gpu_ids:
            env = _build_subprocess_env(gpu_ids)
            try:
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                return 124, "subprocess timeout"
            return proc.returncode, (proc.stdout + proc.stderr)[-4000:]

    return await asyncio.to_thread(_go)


async def _boot_check_app_with_gpu(
    app_path: Path,
    *,
    n_gpus: int = 1,
    wait_s: int = 20,
) -> tuple[bool, str]:
    """Launch app.py with the GPU pool held, wait for the Gradio URL line, kill."""

    def _go() -> tuple[bool, str]:
        with acquire_gpus(n_gpus) as gpu_ids:
            env = _build_subprocess_env(gpu_ids)
            log_path = Path("/tmp") / f"gradio-{app_path.parent.name}.log"
            log_path.write_text("")
            logfile = log_path.open("w")
            proc = subprocess.Popen(
                ["uv", "run", "--project", str(Path.cwd()), "python", str(app_path)],
                env=env,
                stdout=logfile,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.time() + wait_s
                while time.time() < deadline:
                    text = log_path.read_text() if log_path.exists() else ""
                    if "Running on local URL" in text:
                        return True, text[-1000:]
                    if "Traceback" in text or "Error " in text:
                        return False, text[-1000:]
                    time.sleep(0.5)
                return False, (log_path.read_text() if log_path.exists() else "")[-1000:]
            finally:
                with contextlib.suppress(ProcessLookupError):
                    proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                logfile.close()

    return await asyncio.to_thread(_go)


async def _drain(stage: str, slug: str, agen: Any) -> None:
    async for message in agen:
        typer.echo(f"[{stage}/{slug}] {message}")


async def _drain_with_timeout(stage: str, slug: str, agen: Any, timeout_s: int) -> None:
    """Cap a stage's wall-clock; emit `stage_timeout` and raise on overrun."""
    try:
        await asyncio.wait_for(_drain(stage, slug, agen), timeout=timeout_s)
    except TimeoutError:
        emit("stage_timeout", slug=slug, stage=stage, timeout_s=timeout_s)
        raise


async def _completion_with_timeout(
    tier: Tier, prompt: str, *, system_prompt: str, timeout_s: int
) -> str:
    return await asyncio.wait_for(
        run_at_tier_completion(tier, prompt, system_prompt=system_prompt),
        timeout=timeout_s,
    )


def _latest_benchmark(attempt_dir: Path) -> dict[str, Any] | None:
    benchmarks = sorted(attempt_dir.glob("results/*/benchmark.json"))
    if not benchmarks:
        return None
    try:
        data = json.loads(benchmarks[-1].read_text())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


# ---------- pipeline ----------


async def run_pipeline(
    slug: str | None = None,
    *,
    skip_solver: bool = False,
    skip_jury: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Pick → benchmark → review → solve → judge. Returns a status dict."""
    if slug is None:
        block = next_pending()
        if block is None:
            emit("pipeline_idle", reason="no pending blocks")
            return {"status": "idle"}
        slug = block.slug
        block_spec = block.spec
        block_title = block.title
    else:
        # Idempotency: bail early if already graded, unless forced.
        if not force:
            states = load_state()
            existing = states.get(slug)
            if existing and existing.status == "graded":
                emit("pipeline_skipped", slug=slug, reason="already graded")
                return {
                    "status": "skipped",
                    "slug": slug,
                    "verdict_path": existing.verdict_path,
                }
        block_spec = ""
        block_title = slug

    now = datetime.now(UTC).isoformat()
    update_state(slug, status="claimed", claimed_at=now)
    emit("task_claimed", slug=slug, title=block_title)

    goal_dir = Path("experiments") / slug
    goal_dir.mkdir(parents=True, exist_ok=True)

    # 1. PICKER (tier 2 completion) → goal README + benchmark.py
    readme_benchmark = _read_if_exists("README_BENCHMARK.md")
    picker_prompt = f"""\
Slug: {slug}
Title: {block_title}
Spec from BLOCKS.md:
---
{block_spec or "(no spec — infer from title)"}
---

Below is the full construction guide. Follow it exactly.

=== README_BENCHMARK.md ===
{readme_benchmark}
=== END ===

Produce two files:
- `experiments/{slug}/README.md`
- `experiments/{slug}/benchmark.py`
"""
    picker_text = await _completion_with_timeout(
        Tier.STANDARD,
        picker_prompt,
        system_prompt=PICKER_SYSTEM,
        timeout_s=TIER[Tier.STANDARD].wall_clock_s,
    )
    written = apply_blocks(
        picker_text,
        root=Path.cwd(),
        allowed_prefixes=(f"experiments/{slug}/",),
    )
    emit(
        "benchmark_written",
        slug=slug,
        files=[str(p.relative_to(Path.cwd())) for p in written],
    )

    # 2. REVIEWER (tier 1 agentic) → audit + optionally edit
    await _drain_with_timeout(
        "reviewer",
        slug,
        run_at_tier_agentic(
            Tier.EXPERT,
            prompt=f"Audit the benchmark at `experiments/{slug}/` per the system prompt.",
            system_prompt=REVIEWER_SYSTEM,
            allowed_tools=["Read", "Write", "Edit", "Glob", "Grep"],
        ),
        timeout_s=TIER[Tier.EXPERT].wall_clock_s,
    )
    emit("benchmark_reviewed", slug=slug)

    if skip_solver:
        update_state(slug, status="pending_solver")
        emit("pipeline_paused", slug=slug, stage="solver")
        return {"status": "paused", "slug": slug, "stage": "solver"}

    # 3. SOLVER (tier 3 completion) → attempt files; pipeline runs main.py + boot-check.
    update_state(slug, status="solving")
    attempt_name = _suggest_attempt_name(slug)
    attempt_dir = goal_dir / attempt_name
    attempt_dir.mkdir(parents=True, exist_ok=True)
    _ensure_symlink(attempt_dir)

    readme_experiment = _read_if_exists("README_EXPERIMENT.md")
    goal_readme = _read_if_exists(goal_dir / "README.md")
    benchmark_py = _read_if_exists(goal_dir / "benchmark.py")

    gpu_requirement = _load_optional(goal_dir / "benchmark.py", "GPU_REQUIREMENT") or 1
    broken_predicate = _load_optional(goal_dir / "benchmark.py", "is_obviously_broken")

    emit("attempt_started", slug=slug, attempt=attempt_name, gpu_requirement=gpu_requirement)
    solver_prompt = f"""\
Slug: {slug}
Attempt name (already chosen, use it): {attempt_name}

Repo conventions:
=== README_EXPERIMENT.md ===
{readme_experiment}
=== END ===

Goal:
=== experiments/{slug}/README.md ===
{goal_readme}
=== END ===

=== experiments/{slug}/benchmark.py ===
{benchmark_py}
=== END ===

Emit:
- `experiments/{slug}/{attempt_name}/main.py`
- `experiments/{slug}/{attempt_name}/app.py`
- `experiments/{slug}/{attempt_name}/README.md`
"""
    solver_text = await _completion_with_timeout(
        Tier.QUICK,
        solver_prompt,
        system_prompt=SOLVER_SYSTEM,
        timeout_s=TIER[Tier.QUICK].wall_clock_s,
    )
    apply_blocks(
        solver_text,
        root=Path.cwd(),
        allowed_prefixes=(f"experiments/{slug}/{attempt_name}/",),
    )

    # 3a. Pipeline executes main.py and boot-checks app.py with GPU slots held.
    main_path = attempt_dir / "main.py"
    if main_path.exists():
        rc, log = await _run_subprocess_with_gpu(
            ["uv", "run", "--project", str(Path.cwd()), "python", str(main_path)],
            n_gpus=gpu_requirement,
            timeout=600,
        )
        emit(
            "solver_main_run",
            slug=slug,
            attempt=attempt_name,
            returncode=rc,
            log_tail=log[-500:],
        )
        if rc != 0:
            update_state(slug, status="failed")
            emit("pipeline_failed", slug=slug, reason=f"main.py exited {rc}")
            return {"status": "failed", "slug": slug, "stage": "solver_run", "log": log}

    app_path = attempt_dir / "app.py"
    if app_path.exists():
        ok, log = await _boot_check_app_with_gpu(app_path, n_gpus=gpu_requirement)
        emit(
            "solver_app_boot",
            slug=slug,
            attempt=attempt_name,
            ok=ok,
            log_tail=log[-500:],
        )

    emit("attempt_done", slug=slug, attempt=attempt_name)

    # 3b. Short-circuit on obviously-broken metrics — skip the (expensive) jury.
    if broken_predicate is not None:
        bench = _latest_benchmark(attempt_dir)
        metrics = (bench or {}).get("metrics") or {}
        try:
            broken = bool(broken_predicate(metrics))
        except Exception as exc:  # noqa: BLE001 — predicate is goal-author code
            emit("predicate_error", slug=slug, attempt=attempt_name, error=str(exc))
            broken = False
        if broken:
            update_state(slug, status="failed", attempt=attempt_name)
            emit(
                "short_circuit",
                slug=slug,
                attempt=attempt_name,
                reason="is_obviously_broken returned True",
                metrics=metrics,
            )
            return {
                "status": "short_circuit",
                "slug": slug,
                "attempt": attempt_name,
                "reason": "is_obviously_broken",
            }

    if skip_jury:
        update_state(slug, status="awaiting_jury", attempt=attempt_name)
        emit("pipeline_paused", slug=slug, stage="jury")
        return {
            "status": "paused",
            "slug": slug,
            "stage": "jury",
            "attempt": attempt_name,
        }

    # 4. JURY (tier 1 agentic) → verdict.json
    await _drain_with_timeout(
        "jury",
        slug,
        run_at_tier_agentic(
            Tier.EXPERT,
            prompt=(
                f"Grade `experiments/{slug}/{attempt_name}` against the rubric. "
                f"Write `verdict.json` per the system prompt's schema."
            ),
            system_prompt=JURY_SYSTEM,
            allowed_tools=["Read", "Write", "Glob", "Grep"],
        ),
        timeout_s=TIER[Tier.EXPERT].wall_clock_s,
    )
    verdict_path = attempt_dir / "verdict.json"
    update_state(
        slug,
        status="graded",
        attempt=attempt_name,
        verdict_path=str(verdict_path),
    )
    emit(
        "graded",
        slug=slug,
        attempt=attempt_name,
        verdict_path=str(verdict_path),
    )
    return {
        "status": "complete",
        "slug": slug,
        "attempt": attempt_name,
        "verdict": str(verdict_path),
    }


async def run_pipeline_multi(
    *,
    count: int | None = None,
    n_concurrent: int | None = None,
    skip_solver: bool = False,
    skip_jury: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Fan out across pending slugs concurrently.

    The GPU semaphore throttles subprocess execution to the GPU pool size, so
    you can leave `n_concurrent=None` and the LLM API calls run in parallel
    while subprocesses queue. Set `n_concurrent` to rate-limit the LLM side
    (e.g. provider concurrency caps).
    """
    pending = list_pending(count)
    if not pending:
        emit("pipeline_idle", reason="no pending blocks")
        return {"status": "idle"}

    sem = asyncio.Semaphore(n_concurrent) if n_concurrent else None

    async def _one(slug: str) -> Any:
        async def _go() -> Any:
            return await run_pipeline(
                slug=slug, skip_solver=skip_solver, skip_jury=skip_jury, force=force
            )

        if sem is None:
            return await _go()
        async with sem:
            return await _go()

    tasks = [_one(b.slug) for b in pending]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return {
        "status": "complete",
        "count": len(pending),
        "results": [str(r) for r in results],
    }


def run_pipeline_sync(
    slug: str | None = None,
    *,
    skip_solver: bool = False,
    skip_jury: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Convenience wrapper for callers outside an async context."""
    return asyncio.run(
        run_pipeline(slug=slug, skip_solver=skip_solver, skip_jury=skip_jury, force=force)
    )
