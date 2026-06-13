"""Sequential pipeline: pick → smoke-test → review → solve → judge.

Tier wiring (each loop escalates one rung on the tier ladder after `_base`
retries are exhausted):
    task_picker        STANDARD → EXPERT   (completion → agentic)
    benchmark_reviewer EXPERT              (agentic — Opus audits + edits)
    solver             QUICK → STANDARD → EXPERT  (completion → completion → agentic)
    jury               EXPERT              (agentic — Opus writes verdict.json)

Smoke test + feedback loops
---------------------------
After the picker writes the goal's three files, the pipeline runs
`benchmark.score(task.evaluate(task.random_model_fn()))` in a CPU-only
subprocess with a tight timeout (`settings.smoke_test_timeout_s`). If it
crashes, the traceback is folded back into the picker's next prompt — cheaper
than booting the agentic reviewer just to discover the contract is broken.
After `picker_retries_base` failures the picker escalates to EXPERT (agentic
Opus); after `picker_retries_escalated` more failures the slug is marked
failed.

The solver works the same way one tier down: it iterates QUICK → STANDARD →
EXPERT, each retry's prompt carries either the `main.py` traceback or the
degenerate metrics dict (when `is_obviously_broken` returns True) from the prior round.
By the time the jury runs, the solver has already converged on a non-broken
benchmark.

Cost / time controls
--------------------
- Per-tier `wall_clock_s` enforced via `asyncio.wait_for` around each consumer.
- `is_obviously_broken(metrics)` (optional, declared in the goal's benchmark.py)
  drives the solver loop and, after exhaustion, short-circuits the jury.
- `--force` and idempotency: re-running a graded slug exits early unless forced.
- `pipeline-multi` fans out across pending slugs; the GPU semaphore throttles
  the subprocess stages (`main.py`, `app.py` boot-check) to the GPU pool size.
- Subprocesses inherit `HF_HOME` so multi-GB checkpoint downloads aren't repeated.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from agentic import usage
from agentic.blocks import list_pending, load_state, next_pending, update_state
from agentic.config import TIER, Tier, settings
from agentic.events import emit, read_events
from agentic.file_blocks import OUTPUT_CONTRACT, apply_blocks
from agentic.gpu import acquire_gpus
from agentic.runner import run_at_tier_agentic, run_at_tier_completion
from agentic.verdict import JURY_OUTPUT_SCHEMA

# ---------- system prompts (kept byte-stable for prompt caching) ----------

_PICKER_SYSTEM_BASE = """\
You scaffold a new mechanistic-interpretability goal. The goal owns THREE
files; every attempt imports the second and third instead of duplicating
their content:

1. `experiments/<slug>/README.md` — the question, the setup (synthetic vs
   trained-model), the canonical measurement condition, the payload-contract
   table, and the metrics table.
2. `experiments/<slug>/task.py` — the data generator and the evaluator.
   Exports `generate(seed) -> Batch` and `evaluate(model_fn) -> payload`. The
   payload dict it returns must match `benchmark.score()`'s expected shape
   exactly. Two attempts at the same goal must never disagree on the data —
   that's the whole point of factoring this out.

   task.py MUST also export `random_model_fn() -> ModelFn`: a callable with
   exactly the same signature as a real `model_fn`, but whose body returns
   random / zero values of the right shape. Pure NumPy, no torch, no GPU.
   Between the picker and the reviewer the pipeline runs the smoke test
       payload = task.evaluate(task.random_model_fn())
       metrics = benchmark.score(payload)
   and fails the goal back to the picker (with the traceback) if any of
   those three calls crashes. The smoke test exists so the reviewer never
   has to debug shape mismatches by hand — get the contract right and the
   reviewer focuses on substance.

3. `experiments/<slug>/benchmark.py` — exports `VERSION = 1` and
   `score(payload) -> dict[str, float | int]`. One headline summary metric,
   per-slice values, and a baseline. Handles edge cases (empty sweeps, zero
   denominators) explicitly. Optionally exports `GPU_REQUIREMENT: int` and
   `is_obviously_broken(metrics: dict) -> bool`.

You will receive the slug, the title, the spec from BLOCKS.md, and the full
text of README_BENCHMARK.md (the construction guide). Follow that guide.
"""

PICKER_SYSTEM_COMPLETION = _PICKER_SYSTEM_BASE + "\n" + OUTPUT_CONTRACT

PICKER_SYSTEM_AGENTIC = (
    _PICKER_SYSTEM_BASE
    + "\n"
    + """\
Use the Write tool to create the three files at the paths above. Do not
emit `<<FILE: ...>>` blocks; the file system is your output channel. Read
any existing files at those paths first — you may be retrying after the
completion-tier picker left the goal in a broken state.
"""
)

REVIEWER_SYSTEM = """\
You audit a freshly-authored goal (README + task.py + benchmark.py) BEFORE any
attempt is built against it. You have full file tools (Read/Write/Edit/Glob/Grep).

Read:
- `experiments/<slug>/README.md`
- `experiments/<slug>/task.py`
- `experiments/<slug>/benchmark.py`
- `README_BENCHMARK.md`

Check:
- README payload contract matches what `task.evaluate` actually returns and
  what `benchmark.score` actually consumes — no silent shape drift between
  the three files.
- `task.generate` is deterministic for a given seed; the canonical condition
  is unambiguous.
- Metric formulae match what the README claims; headline summary + per-slice
  values + baseline are all present.
- Edge cases (zero denominators, empty sweeps) are handled in `score()`.
- `VERSION = 1` is set; bump procedure documented.

If solid: write one line of approval to `experiments/<slug>/.review.txt`.
If not: edit any of the three files to fix the issues, THEN write the approval
line. The next stage trusts the goal.
"""

_SOLVER_SYSTEM_BASE = """\
You make a first-pass attempt at one mech-interp goal. You cannot execute
code; you only emit files. The pipeline runs them after you.

You will receive: the slug, the chosen attempt_name, the goal's README.md,
the goal's task.py, the goal's benchmark.py, and the README_EXPERIMENT.md
conventions.

Use `task.py`: every attempt imports the data and the evaluator from there.
Your main.py loads it via:

    from agentic.experiments import load_task, record_benchmark, results_dir

    task = load_task(__file__)
    payload = task.evaluate(my_model_fn)  # my_model_fn is THIS attempt's contribution
    record_benchmark(__file__, results_dir(__file__), payload)

For hand-built attempts, `my_model_fn` directly constructs the answer (no
training). For trained attempts, train first, then wrap the trained model in
the function. Either way the same payload shape lands in benchmark.json.

HARD REQUIREMENT — your attempt MUST run on the GPU. The pipeline reserves a
CUDA device for `main.py` and verifies, after it exits, that it actually
allocated CUDA memory; an attempt that runs entirely on the CPU is REJECTED
and you are retried. So `my_model_fn` must do its real compute in torch on
`cuda`, even for hand-built circuits:

    import torch

    DEVICE = "cuda"  # the pipeline guarantees a GPU is visible

    def my_model_fn(q, k):                 # task hands you NumPy arrays
        qt = torch.as_tensor(q, dtype=torch.float32, device=DEVICE)
        kt = torch.as_tensor(k, dtype=torch.float32, device=DEVICE)
        scores = qt @ kt                   # real work on the GPU
        return scores.detach().cpu().numpy()   # task.evaluate expects NumPy back

Hand-set weights are still encouraged (see the rubric) — just express the
circuit as torch tensors on `cuda` instead of NumPy. Do NOT guard the device
behind `torch.cuda.is_available()` fallbacks to CPU; the GPU is guaranteed and
a silent CPU fallback fails the guard.

Emit these files exactly (no others):
- `experiments/<slug>/<attempt_name>/main.py`
- `experiments/<slug>/<attempt_name>/app.py` — Gradio Blocks app with a Demo
  tab and a Benchmark tab that drops in `agentic.experiments.benchmark_panel(<goal_dir>)`.

  Two contracts for `app.py`:
  (a) Expose a module-level `demo: gr.Blocks`. The pipeline boot-checks by
      importing the module and asserting `isinstance(demo, gr.Blocks)` — no
      port binding, so structural bugs (import error, wrong type, etc.)
      get caught instantly.
  (b) Every Gradio call (`.click`, `.change`, `.load`, `.select`, ...) MUST
      live INSIDE the `with gr.Blocks() as demo:` block. Calling any of
      them at module level raises `AttributeError: Cannot call X outside of
      a gradio.Blocks context` and fails the boot-check. The canonical shape:

          with gr.Blocks() as demo:
              ...
              btn.click(fn, inputs=..., outputs=...)
              demo.load(fn, inputs=..., outputs=...)   # inside, not after

          if __name__ == "__main__":
              demo.launch()
- `experiments/<slug>/<attempt_name>/README.md` — two sections: *What I did*
  (3-6 sentences, naming the attempt type — hand_built / trained / interp)
  and *Why this visualisation*.

Do NOT emit the `pyproject.toml` symlink — the pipeline creates it. Do NOT
emit any `results/` files — `main.py` produces those at run time.

"""

SOLVER_SYSTEM = _SOLVER_SYSTEM_BASE + OUTPUT_CONTRACT

SOLVER_SYSTEM_AGENTIC = (
    _SOLVER_SYSTEM_BASE
    + """\
Use the Write tool to create the files at the paths above. Do not emit
`<<FILE: ...>>` blocks; the file system is your output channel. Read any
existing files at those paths first — you may be retrying after a
completion-tier solver left the attempt in a broken state.
"""
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
`overall` is derived from the MEAN of your scored rubric items (0/N-A items
excluded): mean <2 → `fail`, <4 → `borderline`, <5 → `good`, exactly 5 →
`perfect`. Set `overall` to match that mean — the dashboard recomputes it from
your scores, so keep the two consistent.

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
    """Pick the next free attempt folder: first_pass, then pass_2, pass_3, …

    Returns a name that doesn't collide with any existing attempt directory,
    even an empty one a failed solver left behind. A retry that re-enters the
    solver therefore lands in a fresh folder *beside* the previous attempt
    rather than overwriting it, so a failed try is preserved for comparison.
    """
    goal_dir = Path("experiments") / slug
    if not goal_dir.is_dir():
        return "first_pass"
    existing = {d.name for d in goal_dir.iterdir() if d.is_dir()}
    if "first_pass" not in existing:
        return "first_pass"
    n = 2
    while f"pass_{n}" in existing:
        n += 1
    return f"pass_{n}"


def _attempt_order(name: str) -> int:
    """Sort key for attempt folders: first_pass=1, pass_2=2, … (0 = not an attempt)."""
    if name == "first_pass":
        return 1
    if name.startswith("pass_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            return 0
    return 0


def _prior_attempt_context(slug: str, attempt_name: str, goal_dir: Path) -> str:
    """Summarise the previous attempt so a retry can learn from it.

    For a second-or-later attempt (pass_2+), returns a prompt block with the
    jury's `verdict.json` plus the prior `main.py` / `app.py` / `README.md`, so
    the solver can read why the last try fell short and change approach. Returns
    "" for a first attempt or when no prior attempt produced code.
    """
    if not goal_dir.is_dir():
        return ""
    priors = sorted(
        (
            d
            for d in goal_dir.iterdir()
            if d.is_dir() and d.name != attempt_name and _attempt_order(d.name) > 0
        ),
        key=lambda d: _attempt_order(d.name),
        reverse=True,
    )
    prior = next((d for d in priors if (d / "main.py").is_file()), None)
    if prior is None:
        return ""

    verdict = _read_if_exists(prior / "verdict.json")
    main_py = _read_if_exists(prior / "main.py")
    app_py = _read_if_exists(prior / "app.py")
    readme = _read_if_exists(prior / "README.md")

    parts = [
        f"\n\n=== PREVIOUS ATTEMPT ({prior.name}) — LEARN FROM IT, DON'T REPEAT IT ===\n"
        "An earlier attempt at this exact task was graded by the jury and did not "
        "pass. Its verdict and full source are below. Read the verdict's notes, "
        "diagnose why it fell short (e.g. it faked the mechanism, the app didn't "
        "boot, weak faithfulness evidence), and take a MEANINGFULLY DIFFERENT "
        "approach.\n"
        "This is a FRESH attempt in its own folder: you are free to DISCARD ALL of "
        "the previous code and start from a clean slate. You are NOT required to "
        "build on it or keep any of it. Treat the prior attempt as a worked example "
        "of what failed, not as a starting point — reuse a piece only if it is "
        "genuinely sound and addresses the verdict. If the root cause is the overall "
        "approach, throw it out entirely and write something new.\n"
    ]
    if verdict:
        parts.append(f"--- {prior.name}/verdict.json (jury) ---\n{verdict}\n")
    if main_py:
        parts.append(f"--- {prior.name}/main.py ---\n{main_py}\n")
    if app_py:
        parts.append(f"--- {prior.name}/app.py ---\n{app_py}\n")
    if readme:
        parts.append(f"--- {prior.name}/README.md ---\n{readme}\n")
    parts.append("=== END PREVIOUS ATTEMPT ===\n")
    return "\n".join(parts)


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


# Uses __APP_PATH__ as a placeholder (replaced with a repr'd path) rather than
# str.format so the body can use braces (dict/set literals) freely.
_APP_IMPORT_CHECK = r'''
import importlib.util, sys, traceback, types
import gradio as gr

spec = importlib.util.spec_from_file_location("app", __APP_PATH__)
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
sys.modules["app"] = m
spec.loader.exec_module(m)
demo = getattr(m, "demo", None)
assert isinstance(demo, gr.Blocks), (
    "app.py must expose a module-level `demo: gr.Blocks`; got %s" % type(demo).__name__
)


def _example_for(c):
    """Best-effort raw (un-preprocessed) example value for an input component."""
    v = getattr(c, "value", None)
    if v is not None:
        return v
    if isinstance(c, gr.Slider):
        return getattr(c, "minimum", 0) or 0
    if isinstance(c, gr.Number):
        return 0
    if isinstance(c, gr.Checkbox):
        return False
    if isinstance(c, (gr.Dropdown, gr.Radio)):
        choices = getattr(c, "choices", None) or []
        if choices:
            first = choices[0]
            return first[1] if isinstance(first, (list, tuple)) else first
        return None
    if isinstance(c, gr.Textbox):
        return ""
    return None


def _exhaust(result):
    """Streaming handlers return a generator; pull one value to exercise the body."""
    if isinstance(result, types.GeneratorType):
        for _ in result:
            break


# Run every event handler once with example inputs. A handler that raises is a
# real bug (NameError from a missing import, bad indexing, Gradio misuse) that a
# bare import check would miss because the callbacks never fire until clicked.
ran = 0
errors = []
fns = demo.fns.values() if isinstance(demo.fns, dict) else list(demo.fns)
for bf in fns:
    fn = getattr(bf, "fn", None)
    if fn is None or not callable(fn):
        continue
    inputs = list(getattr(bf, "inputs", None) or [])
    examples = [_example_for(c) for c in inputs]
    label = getattr(bf, "api_name", None) or getattr(fn, "__name__", repr(fn))
    try:
        if getattr(bf, "inputs_as_dict", False):
            _exhaust(fn({c: v for c, v in zip(inputs, examples)}))
        else:
            _exhaust(fn(*examples))
        ran += 1
    except Exception:
        errors.append("event handler %r failed:\n%s" % (label, traceback.format_exc()))

if errors:
    raise SystemExit("\n\n".join(errors))
print("ok (ran %d handler(s))" % ran)
'''


async def _boot_check_app_with_gpu(
    app_path: Path,
    *,
    n_gpus: int = 1,
    wait_s: int = 120,
) -> tuple[bool, str]:
    """Verify `app.py` imports cleanly, exposes `demo: gr.Blocks`, and that every
    event handler runs once with example inputs.

    Cheaper than launching the server: catches Gradio API misuse (calling
    `.load`/`.click`/etc. outside a Blocks context), import errors, syntax
    errors, missing `demo`, and wrong type — all the bugs a launch would
    surface in its first second, but with no port binding and no wait for
    the "Running on local URL" line.

    Beyond importing, it enumerates `demo.fns` and invokes each handler's `fn`
    once with example inputs derived from its input components (a generator
    result is pulled once). This exercises callback bodies that a bare import
    would never run, surfacing the bugs that only fire on interaction — e.g. a
    handler referencing `np` when `app.py` never imported numpy.
    """

    def _go() -> tuple[bool, str]:
        with acquire_gpus(n_gpus) as gpu_ids:
            env = _build_subprocess_env(gpu_ids)
            script = _APP_IMPORT_CHECK.replace("__APP_PATH__", repr(str(app_path)))
            try:
                proc = subprocess.run(
                    ["uv", "run", "--project", str(Path.cwd()), "python", "-c", script],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=wait_s,
                )
            except subprocess.TimeoutExpired:
                return False, f"app import check timed out after {wait_s}s"
            ok = proc.returncode == 0 and "ok" in proc.stdout
            return ok, (proc.stdout + proc.stderr)[-1500:]

    return await asyncio.to_thread(_go)


def _summarise_message(message: Any) -> str:
    """Compact one-line summary of an SDK message.

    Full message reprs include thinking signatures and usage dicts that can
    each be many KB; dumping them at every message blows up stdout pipes.
    """
    name = type(message).__name__
    parts: list[str] = [name]
    if hasattr(message, "subtype") and message.subtype is not None:
        parts.append(f"subtype={message.subtype}")
    content = getattr(message, "content", None)
    if isinstance(content, list):
        kinds = [type(b).__name__ for b in content]
        parts.append(f"blocks={','.join(kinds) or '-'}")
        for b in content:
            text = getattr(b, "text", None)
            if isinstance(text, str) and text:
                snippet = text.strip().splitlines()[0][:120]
                parts.append(f'text="{snippet}"')
                break
            tool_name = getattr(b, "name", None)
            if tool_name:
                parts.append(f"tool={tool_name}")
                break
    return " ".join(parts)


async def _drain(stage: str, slug: str, agen: Any) -> None:
    async for message in agen:
        try:
            typer.echo(f"[{stage}/{slug}] {_summarise_message(message)}")
        except BlockingIOError:
            # Pipe consumer (tee, terminal) fell behind. Don't kill the pipeline
            # — the event log + state files are the durable record.
            continue


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


# ---------- smoke test (picker → reviewer) ----------

_SMOKE_TEST_SCRIPT = """\
import json, sys, traceback
import importlib.util
import os.path

SLUG_DIR = sys.argv[1]


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SLUG_DIR, fname))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {fname}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    task = _load("_smoke_task", "task.py")
    benchmark = _load("_smoke_benchmark", "benchmark.py")
    if not hasattr(task, "random_model_fn"):
        raise AttributeError(
            "task.py is missing `random_model_fn() -> ModelFn`. The pipeline "
            "needs it for the post-picker smoke test."
        )
    model_fn = task.random_model_fn()
    payload = task.evaluate(model_fn)
    metrics = benchmark.score(payload)
    print("SMOKE_OK")
    print(json.dumps(metrics, default=str))
except Exception:
    traceback.print_exc()
    sys.exit(1)
"""


async def _smoke_test_benchmark(slug: str, *, timeout_s: int) -> tuple[bool, str]:
    """Sub-second contract check: task.evaluate(random_model_fn) → benchmark.score.

    Runs in a CPU-only subprocess (no GPU acquire, no torch import).
    Returns (ok, output). On failure `output` is the traceback we feed back
    to the picker so it can fix the issue without burning a reviewer turn.
    """
    slug_dir = (Path("experiments") / slug).resolve()
    script_path = Path("/tmp") / f"agentic_smoke_{slug}.py"
    script_path.write_text(_SMOKE_TEST_SCRIPT)

    def _go() -> tuple[bool, str]:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        if settings.hf_home:
            env["HF_HOME"] = settings.hf_home
        try:
            proc = subprocess.run(
                [
                    "uv",
                    "run",
                    "--project",
                    str(Path.cwd()),
                    "python",
                    str(script_path),
                    str(slug_dir),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, f"smoke test timed out after {timeout_s}s"
        output = (proc.stdout + proc.stderr)[-4000:]
        ok = proc.returncode == 0 and "SMOKE_OK" in proc.stdout
        return ok, output

    return await asyncio.to_thread(_go)


# ---------- retry loops with tier escalation ----------

# Ascending rung order for the solver/picker tier ladder. Used to honour a
# `min_tier` floor: a re-run can skip the cheaper rungs and go straight to a
# better model (see `run_pipeline`'s `min_tier`).
_TIER_RANK: dict[Tier, int] = {Tier.QUICK: 0, Tier.STANDARD: 1, Tier.EXPERT: 2}


def _build_picker_prompt(
    slug: str,
    block_title: str,
    block_spec: str,
    readme_benchmark: str,
    feedback: str,
) -> str:
    return f"""\
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
{feedback}
Produce three files:
- `experiments/{slug}/README.md`
- `experiments/{slug}/task.py`        (must export `generate`, `evaluate`, `random_model_fn`)
- `experiments/{slug}/benchmark.py`
"""


async def _picker_with_smoke_retries(
    slug: str,
    block_title: str,
    block_spec: str,
    readme_benchmark: str,
) -> tuple[bool, str]:
    """PICKER → smoke test loop with tier escalation.

    Runs the picker at STANDARD for `picker_retries_base` attempts. On each
    failure the smoke-test traceback is appended to the next prompt as
    feedback. After exhaustion escalates to EXPERT (agentic Opus) for
    `picker_retries_escalated` more attempts. Returns (ok, last_smoke_log).
    """
    feedback = ""
    n_base = settings.picker_retries_base
    n_esc = settings.picker_retries_escalated
    last_log = ""

    for i in range(n_base + n_esc):
        tier = Tier.STANDARD if i < n_base else Tier.EXPERT
        prompt = _build_picker_prompt(slug, block_title, block_spec, readme_benchmark, feedback)
        cfg = TIER[tier]

        emit("picker_attempt", slug=slug, attempt_idx=i, tier=tier.value)

        try:
            if cfg.mode == "completion":
                text = await _completion_with_timeout(
                    tier,
                    prompt,
                    system_prompt=PICKER_SYSTEM_COMPLETION,
                    timeout_s=cfg.wall_clock_s,
                )
                written = apply_blocks(
                    text,
                    root=Path.cwd(),
                    allowed_prefixes=(f"experiments/{slug}/",),
                )
                emit(
                    "benchmark_written",
                    slug=slug,
                    tier=tier.value,
                    attempt_idx=i,
                    files=[str(p.relative_to(Path.cwd())) for p in written],
                )
            else:
                await _drain_with_timeout(
                    "picker",
                    slug,
                    run_at_tier_agentic(
                        tier,
                        prompt=prompt,
                        system_prompt=PICKER_SYSTEM_AGENTIC,
                        allowed_tools=["Read", "Write", "Edit"],
                    ),
                    timeout_s=cfg.wall_clock_s,
                )
                emit("benchmark_written", slug=slug, tier=tier.value, attempt_idx=i)
        except Exception as exc:  # noqa: BLE001 — feed any failure into the next retry
            last_log = f"{type(exc).__name__}: {exc}"
            emit(
                "picker_call_error",
                slug=slug,
                attempt_idx=i,
                tier=tier.value,
                error=last_log,
            )
            feedback = (
                "\n\n=== PREVIOUS ATTEMPT'S MODEL CALL FAILED ===\n"
                f"{last_log}\n"
                "=== END ===\n\n"
                "Try again — keep your response short and the file blocks "
                "complete. If the failure was a timeout, reduce verbosity."
            )
            continue

        ok, log = await _smoke_test_benchmark(slug, timeout_s=settings.smoke_test_timeout_s)
        last_log = log
        emit(
            "benchmark_smoke",
            slug=slug,
            attempt_idx=i,
            tier=tier.value,
            ok=ok,
            log_tail=log[-500:],
        )
        if ok:
            return True, log

        feedback = (
            "\n\n=== PREVIOUS ATTEMPT FAILED THE SMOKE TEST ===\n"
            "The pipeline ran:\n"
            "    payload = task.evaluate(task.random_model_fn())\n"
            "    metrics = benchmark.score(payload)\n"
            "and got:\n\n"
            f"{log}\n"
            "=== END ===\n\n"
            "Re-emit ALL THREE files (README.md, task.py, benchmark.py) with "
            "the fix. Common causes:\n"
            "- `random_model_fn` missing, or its return signature doesn't match "
            "what `task.evaluate` calls.\n"
            "- payload dict from `task.evaluate` doesn't match what "
            "`benchmark.score` consumes (key names, sweep length, missing "
            "`version`).\n"
            "- import-time error in task.py or benchmark.py (typo, missing "
            "import, wrong `VERSION`).\n"
        )

    return False, last_log


def _build_solver_prompt(
    slug: str,
    attempt_name: str,
    readme_experiment: str,
    goal_readme: str,
    task_py: str,
    benchmark_py: str,
    feedback: str,
    prior_context: str = "",
) -> str:
    return f"""\
Slug: {slug}
Attempt name (already chosen, use it): {attempt_name}
{prior_context}

Repo conventions:
=== README_EXPERIMENT.md ===
{readme_experiment}
=== END ===

Goal:
=== experiments/{slug}/README.md ===
{goal_readme}
=== END ===

=== experiments/{slug}/task.py ===
{task_py}
=== END ===

=== experiments/{slug}/benchmark.py ===
{benchmark_py}
=== END ===

CRITICAL — derive `model_fn`'s signature from `task.py` above, do NOT assume one:

1. Find the `ModelFn = Callable[...]` type alias in `task.py`. That gives the
   exact argument types and return type your `model_fn` must satisfy.
2. Find the line where `task.evaluate` invokes `model_fn(...)`. The arguments
   it passes there are EXACTLY what your function will receive — same order,
   same count. Some goals pass `model_fn(batch)`; others pass unpacked fields
   like `model_fn(batch.q_A, batch.q_B, ...)`. Match the goal's choice.
3. `Batch` is a frozen dataclass — access fields as `batch.foo`, NOT `batch["foo"]`.
4. Return one numpy array with the exact shape `task.evaluate` validates against
   (look for `if logits.shape != (...)` or similar in `task.py`).

A signature mismatch crashes immediately on `task.evaluate(model_fn)`. The
pipeline will retry you with the traceback, but get it right first.
{feedback}
Emit:
- `experiments/{slug}/{attempt_name}/main.py`
- `experiments/{slug}/{attempt_name}/app.py`
- `experiments/{slug}/{attempt_name}/README.md`
"""


async def _solver_with_benchmark_retries(
    slug: str,
    attempt_name: str,
    goal_dir: Path,
    attempt_dir: Path,
    gpu_requirement: int,
    broken_predicate: Any,
    min_tier: Tier | None = None,
) -> tuple[bool, dict[str, Any] | None, str]:
    """SOLVER → run main.py → check benchmark loop with tier escalation.

    Each iteration:
        1. SOLVER emits attempt files (prompt includes benchmark feedback
           from prior rounds).
        2. Pipeline runs main.py. If it crashes, feedback = traceback, retry.
        3. If `is_obviously_broken(metrics)` is True, feedback = the metrics
           dict, retry. The solver sees its own benchmark output before the
           jury ever does.
        4. Boot-check app.py (import + module-level `demo: gr.Blocks`). If it's
           missing or fails to import, feedback = the boot error, retry — so a
           broken Gradio app is fixed in-loop, not just flagged by the jury.
        5. Otherwise success — return (True, metrics, log).

    Tries QUICK tier `solver_retries_base` times, escalates to STANDARD for
    `solver_retries_escalated` more, then to EXPERT (agentic Opus) for
    `solver_retries_expert` more. Returns (ok, last_metrics_or_None, last_log).

    `min_tier` drops every rung below it from the schedule, so a re-run can go
    straight to a better model (e.g. EXPERT-only) instead of burning the cheap
    QUICK/STANDARD attempts again.
    """
    readme_experiment = _read_if_exists("README_EXPERIMENT.md")
    goal_readme = _read_if_exists(goal_dir / "README.md")
    task_py = _read_if_exists(goal_dir / "task.py")
    benchmark_py = _read_if_exists(goal_dir / "benchmark.py")
    # On a retry (pass_2+), carry the previous attempt's verdict + code into the
    # prompt so the solver learns from the prior failure instead of re-deriving
    # the same dead end. Constant across this attempt's own retry iterations.
    prior_context = _prior_attempt_context(slug, attempt_name, goal_dir)

    feedback = ""
    n_base = settings.solver_retries_base
    n_esc = settings.solver_retries_escalated
    n_expert = settings.solver_retries_expert
    # Flooring the run to EXPERT drops the cheap rungs below, so the expert tier
    # is the only fallback left — bump its retry budget instead of the usual
    # last-ditch 2 so the chosen tier actually gets a real shot.
    if min_tier is Tier.EXPERT:
        n_expert = settings.solver_retries_expert_floored
    last_log = ""
    last_metrics: dict[str, Any] | None = None

    # Full escalation ladder, then drop rungs below the requested floor.
    schedule = [Tier.QUICK] * n_base + [Tier.STANDARD] * n_esc + [Tier.EXPERT] * n_expert
    if min_tier is not None:
        schedule = [t for t in schedule if _TIER_RANK[t] >= _TIER_RANK[min_tier]]
    if not schedule:
        emit("solver_no_attempts", slug=slug, attempt=attempt_name, min_tier=getattr(min_tier, "value", None))
        return False, None, "no solver attempts scheduled (min_tier above all configured retries)"

    for i, tier in enumerate(schedule):
        prompt = _build_solver_prompt(
            slug,
            attempt_name,
            readme_experiment,
            goal_readme,
            task_py,
            benchmark_py,
            feedback,
            prior_context,
        )
        cfg = TIER[tier]

        emit("solver_attempt", slug=slug, attempt=attempt_name, attempt_idx=i, tier=tier.value)

        try:
            if cfg.mode == "completion":
                text = await _completion_with_timeout(
                    tier, prompt, system_prompt=SOLVER_SYSTEM, timeout_s=cfg.wall_clock_s
                )
                # Completion tiers emit file blocks; parse and write them. Agentic
                # tiers (EXPERT) write the files directly via the Write tool.
                apply_blocks(
                    text,
                    root=Path.cwd(),
                    allowed_prefixes=(f"experiments/{slug}/{attempt_name}/",),
                )
            else:
                await _drain_with_timeout(
                    "solver",
                    slug,
                    run_at_tier_agentic(
                        tier,
                        prompt=prompt,
                        system_prompt=SOLVER_SYSTEM_AGENTIC,
                        allowed_tools=["Read", "Write", "Edit"],
                    ),
                    timeout_s=cfg.wall_clock_s,
                )
        except Exception as exc:  # noqa: BLE001 — feed any failure into the next retry
            last_log = f"{type(exc).__name__}: {exc}"
            emit(
                "solver_call_error",
                slug=slug,
                attempt=attempt_name,
                attempt_idx=i,
                tier=tier.value,
                error=last_log,
            )
            feedback = (
                "\n\n=== PREVIOUS ATTEMPT'S MODEL CALL FAILED ===\n"
                f"{last_log}\n"
                "=== END ===\n\n"
                "Try again — keep your response short and the file blocks "
                "complete. If the failure was a timeout, reduce verbosity."
            )
            continue

        main_path = attempt_dir / "main.py"
        if not main_path.exists():
            feedback = (
                "\n\n=== PREVIOUS ATTEMPT DID NOT EMIT main.py ===\n"
                "Re-emit ALL THREE files: main.py, app.py, README.md.\n"
            )
            continue

        # Launch through the GPU guard, not `python main.py` directly: it runs
        # the attempt and then asserts it actually allocated CUDA memory, so a
        # pure-CPU/NumPy attempt fails loudly instead of wasting a reserved slot.
        rc, log = await _run_subprocess_with_gpu(
            [
                "uv", "run", "--project", str(Path.cwd()),
                "python", "-m", "agentic.gpu_guard", str(main_path),
            ],
            n_gpus=gpu_requirement,
            timeout=600,
        )
        last_log = log
        emit(
            "solver_main_run",
            slug=slug,
            attempt=attempt_name,
            attempt_idx=i,
            tier=tier.value,
            returncode=rc,
            log_tail=log[-500:],
        )

        if rc != 0:
            feedback = (
                "\n\n=== PREVIOUS ATTEMPT'S main.py CRASHED ===\n"
                f"Exit code: {rc}\n"
                "Tail of stdout+stderr:\n"
                f"{log}\n"
                "=== END ===\n\n"
                "Re-emit ALL THREE files with the fix. Common causes:\n"
                "- `model_fn` signature doesn't match what `task.evaluate` "
                "passes (check the call site in task.py).\n"
                "- Wrong return shape or dtype from `model_fn`.\n"
                "- Import errors / typos.\n"
            )
            continue

        bench = _latest_benchmark(attempt_dir)
        metrics = (bench or {}).get("metrics") or {}
        last_metrics = metrics

        if broken_predicate is not None:
            try:
                broken = bool(broken_predicate(metrics))
            except Exception as exc:  # noqa: BLE001 — predicate is goal-author code
                emit("predicate_error", slug=slug, attempt=attempt_name, error=str(exc))
                broken = False
            if broken:
                feedback = (
                    "\n\n=== PREVIOUS ATTEMPT'S BENCHMARK IS DEGENERATE ===\n"
                    "main.py ran cleanly, but `is_obviously_broken(metrics)` "
                    "returned True. The metrics it judged:\n\n"
                    f"{json.dumps(metrics, indent=2, default=str)}\n\n"
                    "=== END ===\n\n"
                    "Re-emit ALL THREE files with a substantively different "
                    "approach — the previous strategy isn't beating the floor "
                    "predicate.\n"
                )
                continue

        # app.py must boot, just as main.py must run. A non-booting Gradio app
        # feeds its import/Blocks error back as feedback and retries, instead of
        # being silently committed and only flagged by the jury at grading time.
        app_path = attempt_dir / "app.py"
        if not app_path.exists():
            feedback = (
                "\n\n=== PREVIOUS ATTEMPT DID NOT EMIT app.py ===\n"
                "Re-emit ALL THREE files: main.py, app.py, README.md.\n"
            )
            continue

        ok_app, log_app = await _boot_check_app_with_gpu(app_path, n_gpus=gpu_requirement)
        emit(
            "solver_app_boot",
            slug=slug,
            attempt=attempt_name,
            attempt_idx=i,
            tier=tier.value,
            ok=ok_app,
            log_tail=log_app[-500:],
        )
        if not ok_app:
            last_log = log_app
            feedback = (
                "\n\n=== PREVIOUS ATTEMPT'S app.py FAILED ITS BOOT CHECK ===\n"
                "main.py ran and the benchmark passed, but app.py failed its "
                "boot check. The check imports app.py, verifies it exposes a "
                "module-level `demo: gr.Blocks`, and then runs every event "
                "handler once with example inputs derived from its input "
                "components — so a callback that crashes on interaction (not just "
                "an import error) also fails the check.\n"
                "Error tail:\n"
                f"{log_app}\n"
                "=== END ===\n\n"
                "Re-emit ALL THREE files with app.py fixed. Common causes:\n"
                "- Calling `.load`/`.click`/component constructors outside a "
                "`with gr.Blocks() as demo:` context.\n"
                "- Using Gradio APIs that don't exist (e.g. `gr.TabsItem`, "
                "`gr.Tabs(items=...)`, positional data args to `gr.Plot`).\n"
                "- Import errors / typos (e.g. missing `import numpy as np`, or "
                "importing symbols that aren't exported by `agentic`).\n"
                "- Calling `demo.launch()` at module import time.\n"
                "- Not exposing a module-level `demo: gr.Blocks`.\n"
            )
            continue

        return True, metrics, log

    return False, last_metrics, last_log


# ---------- pipeline ----------


def _completed_stages(slug: str) -> tuple[bool, bool]:
    """Inspect the event log for a slug's already-passed prep stages.

    Returns ``(picker_done, reviewer_done)``:

    - ``picker_done`` — a `benchmark_smoke` event recorded ``ok=True``, i.e. the
      picker produced a benchmark that passed the smoke test.
    - ``reviewer_done`` — a `benchmark_reviewed` event was emitted. The reviewer
      only runs after a successful picker, so this implies ``picker_done`` too.

    Used by resume (see `run_pipeline`) to skip prep stages on a retry. Events
    accumulate across runs, so a benchmark that passed on an earlier attempt
    still counts here even after a later solver/jury failure.
    """
    picker_done = reviewer_done = False
    for ev in read_events():
        if ev.get("slug") != slug:
            continue
        etype = ev.get("type")
        if etype == "benchmark_smoke" and ev.get("ok"):
            picker_done = True
        elif etype == "benchmark_reviewed":
            picker_done = reviewer_done = True
    return picker_done, reviewer_done


async def run_pipeline(
    slug: str | None = None,
    *,
    skip_solver: bool = False,
    skip_jury: bool = False,
    force: bool = False,
    resume: bool = False,
    min_tier: Tier | None = None,
) -> dict[str, Any]:
    """Pick → benchmark → review → solve → judge. Returns a status dict.

    `resume` (slug runs only) skips the picker and/or reviewer when a prior run
    already passed them — a benchmark that passed smoke is reused as-is, and a
    benchmark that was already reviewed isn't re-reviewed. The solver onward
    always re-runs. Used to retry a failed task without redoing benchmark prep.

    `min_tier` raises the solver's starting rung: the QUICK/STANDARD attempts
    below it are dropped so a re-run goes straight to a better model. Pairs with
    `resume` for a cheap "retry this at a higher tier only" re-run.
    """
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

    # On resume, skip prep stages a prior run already passed. The picker is only
    # skippable if its benchmark is still on disk for the solver to consume.
    picker_done, reviewer_done = _completed_stages(slug) if resume else (False, False)
    skip_picker = picker_done and (goal_dir / "benchmark.py").exists()

    # All model calls inside this block tally into a per-slug usage tracker.
    # On exit (success or exception) the tracker emits one `usage_summary`
    # event with per-model and total token + cost rollups.
    with usage.track(slug):
        # 1. PICKER → smoke test → retry loop with tier escalation.
        if skip_picker:
            emit("stage_skipped", slug=slug, stage="picker", reason="resume_benchmark_passed")
        else:
            readme_benchmark = _read_if_exists("README_BENCHMARK.md")
            with usage.stage("picker"):
                ok, smoke_log = await _picker_with_smoke_retries(
                    slug, block_title, block_spec, readme_benchmark
                )
            if not ok:
                update_state(slug, status="failed")
                emit(
                    "pipeline_failed",
                    slug=slug,
                    stage="picker",
                    reason="smoke_test_exhausted",
                    log_tail=smoke_log[-500:],
                )
                return {
                    "status": "failed",
                    "slug": slug,
                    "stage": "picker",
                    "reason": "smoke_test_exhausted",
                    "log": smoke_log,
                }

        # 2. REVIEWER (tier 1 agentic) → audit + optionally edit a smoke-tested goal.
        # Skip only when we reused an already-reviewed benchmark; a freshly
        # (re)generated benchmark must always be reviewed.
        if skip_picker and reviewer_done:
            emit("stage_skipped", slug=slug, stage="reviewer", reason="resume_already_reviewed")
        else:
            with usage.stage("reviewer"):
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

        # 3. SOLVER → main.py → benchmark loop with tier escalation.
        update_state(slug, status="solving")
        attempt_name = _suggest_attempt_name(slug)
        attempt_dir = goal_dir / attempt_name
        attempt_dir.mkdir(parents=True, exist_ok=True)
        _ensure_symlink(attempt_dir)

        # Every attempt must run on the GPU (the guard enforces it at runtime),
        # so clamp to a minimum of one slot regardless of what benchmark.py says.
        gpu_requirement = max(1, int(_load_optional(goal_dir / "benchmark.py", "GPU_REQUIREMENT") or 1))
        broken_predicate = _load_optional(goal_dir / "benchmark.py", "is_obviously_broken")

        emit("attempt_started", slug=slug, attempt=attempt_name, gpu_requirement=gpu_requirement)

        with usage.stage("solver"):
            ok, metrics, log = await _solver_with_benchmark_retries(
                slug,
                attempt_name,
                goal_dir,
                attempt_dir,
                gpu_requirement,
                broken_predicate,
                min_tier=min_tier,
            )
        if not ok:
            update_state(slug, status="failed", attempt=attempt_name)
            if metrics is None:
                emit(
                    "pipeline_failed",
                    slug=slug,
                    attempt=attempt_name,
                    stage="solver_run",
                    reason="main_py_exhausted",
                    log_tail=log[-500:],
                )
                return {
                    "status": "failed",
                    "slug": slug,
                    "stage": "solver_run",
                    "reason": "main_py_exhausted",
                    "log": log,
                }
            emit(
                "short_circuit",
                slug=slug,
                attempt=attempt_name,
                reason="is_obviously_broken_exhausted",
                metrics=metrics,
            )
            return {
                "status": "short_circuit",
                "slug": slug,
                "attempt": attempt_name,
                "reason": "is_obviously_broken_exhausted",
            }

        # The Gradio app already passed its boot-check inside the solver loop
        # (a non-booting app.py is a retry condition there), so on a successful
        # solve we know app.py imports and exposes `demo: gr.Blocks`.
        emit("attempt_done", slug=slug, attempt=attempt_name, metrics=metrics)

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
        with usage.stage("jury"):
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
