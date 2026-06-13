"""Pipeline plumbing tests — no LLM calls. Exercises the non-network paths:
blocks parsing, state JSONL, events JSONL, file-block parser, verdict schema."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point AGENTIC_STATE_DIR + AGENTIC_BLOCKS_FILE at a temp dir for each test."""
    state_dir = tmp_path / "state"
    blocks_file = tmp_path / "BLOCKS.md"
    monkeypatch.setenv("AGENTIC_STATE_DIR", str(state_dir))
    monkeypatch.setenv("AGENTIC_BLOCKS_FILE", str(blocks_file))
    # Reload settings to pick up the env override.
    import importlib

    import agentic.config

    importlib.reload(agentic.config)
    return tmp_path


def test_blocks_parses_and_picks_next(tmp_path: Path) -> None:
    blocks_path = Path(os.environ["AGENTIC_BLOCKS_FILE"])
    blocks_path.write_text(
        """# Problems

## Tier 0 — primitives

1. **First task** — `first_one`
   - I/O: spec for first.
   - What makes it hard: nothing.

2. **Second task** — `second_one`
   - I/O: spec for second.
   - builds on: `first_one`.

## How to use this list

Prose with no numbered items — must yield no blocks.
"""
    )
    import importlib

    import agentic.blocks

    importlib.reload(agentic.blocks)

    blocks = agentic.blocks.parse_blocks()
    assert [b.slug for b in blocks] == ["first_one", "second_one"]
    assert blocks[0].title == "First task"
    # Spec captures the indented body; the `first_one` mention in item 2's
    # "builds on" line must not be parsed as a third task.
    assert "spec for first" in blocks[0].spec.lower()

    nxt = agentic.blocks.next_pending()
    assert nxt is not None and nxt.slug == "first_one"

    agentic.blocks.update_state("first_one", status="graded")
    nxt = agentic.blocks.next_pending()
    assert nxt is not None and nxt.slug == "second_one"


def test_state_round_trip() -> None:
    import importlib

    import agentic.blocks

    importlib.reload(agentic.blocks)
    agentic.blocks.update_state("alpha", status="claimed", attempt="a1")
    states = agentic.blocks.load_state()
    assert states["alpha"].status == "claimed"
    assert states["alpha"].attempt == "a1"
    assert states["alpha"].updated_at is not None


def test_events_round_trip() -> None:
    import importlib

    import agentic.events

    importlib.reload(agentic.events)
    agentic.events.emit("task_claimed", slug="x")
    agentic.events.emit("graded", slug="x", attempt="a1")
    events = agentic.events.read_events()
    assert [e["type"] for e in events] == ["task_claimed", "graded"]
    assert events[1]["slug"] == "x"


def test_completed_stages_reads_event_log() -> None:
    """Resume detection: picker_done needs a smoke-ok event, reviewer_done a
    benchmark_reviewed event; an ok=False smoke alone leaves the picker undone."""
    import importlib

    import agentic.events

    importlib.reload(agentic.events)
    from agentic.pipeline import _completed_stages

    # Unknown slug: nothing passed.
    assert _completed_stages("ghost") == (False, False)

    # Picker passed smoke but was never reviewed.
    agentic.events.emit("benchmark_smoke", slug="alpha", ok=True)
    assert _completed_stages("alpha") == (True, False)

    # A failed smoke must not count as picker_done.
    agentic.events.emit("benchmark_smoke", slug="beta", ok=False)
    assert _completed_stages("beta") == (False, False)

    # A review implies the picker passed too, even without a smoke event here.
    agentic.events.emit("benchmark_reviewed", slug="gamma")
    assert _completed_stages("gamma") == (True, True)


def test_suggest_attempt_name_never_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each call picks a fresh folder beside existing attempts, even an empty
    one a failed solver left behind, so a retry never clobbers a prior try."""
    monkeypatch.chdir(tmp_path)
    from agentic.pipeline import _suggest_attempt_name

    goal = tmp_path / "experiments" / "demo"
    goal.mkdir(parents=True)
    assert _suggest_attempt_name("demo") == "first_pass"

    # A failed solver leaves an empty first_pass dir (no main.py): retry must
    # advance to pass_2 rather than overwrite it.
    (goal / "first_pass").mkdir()
    (goal / "__pycache__").mkdir()  # non-attempt dir must be ignored
    assert _suggest_attempt_name("demo") == "pass_2"

    (goal / "pass_2").mkdir()
    assert _suggest_attempt_name("demo") == "pass_3"


def test_file_blocks_parses_and_applies(tmp_path: Path) -> None:
    from agentic.file_blocks import apply_blocks, parse_blocks

    text = """<<FILE: experiments/foo/README.md>>
# Hello

This is a body.
<<END FILE>>

<<FILE: experiments/foo/benchmark.py>>
VERSION = 1


def score(payload):
    return {"version": 1, "x": 1.0}
<<END FILE>>
"""
    pairs = parse_blocks(text)
    assert [p for p, _ in pairs] == [
        "experiments/foo/README.md",
        "experiments/foo/benchmark.py",
    ]

    written = apply_blocks(text, root=tmp_path, allowed_prefixes=("experiments/foo/",))
    assert {p.relative_to(tmp_path).as_posix() for p in written} == {
        "experiments/foo/README.md",
        "experiments/foo/benchmark.py",
    }
    assert (tmp_path / "experiments/foo/README.md").read_text().startswith("# Hello")


def test_file_blocks_rejects_path_escape(tmp_path: Path) -> None:
    from agentic.file_blocks import apply_blocks

    text = """<<FILE: ../escape.txt>>
nope
<<END FILE>>

<<FILE: /etc/passwd>>
nope
<<END FILE>>
"""
    written = apply_blocks(text, root=tmp_path)
    assert written == []
    assert not (tmp_path.parent / "escape.txt").exists()


def test_gpu_pool_acquires_and_releases(tmp_path: Path) -> None:
    import importlib

    import agentic.config
    import agentic.gpu

    importlib.reload(agentic.config)
    importlib.reload(agentic.gpu)

    # Acquire/release one slot, then re-acquire both.
    with agentic.gpu.acquire_gpus(1) as ids:
        assert ids == [0]
    with agentic.gpu.acquire_gpus(2) as ids:
        assert ids == [0, 1]


def test_gpu_pool_rejects_overrequest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTIC_GPU_COUNT", "2")
    import importlib

    import agentic.config
    import agentic.gpu

    importlib.reload(agentic.config)
    importlib.reload(agentic.gpu)

    with pytest.raises(ValueError, match="GPUs but pool has"), agentic.gpu.acquire_gpus(3):
        pass


def test_list_pending_respects_limit() -> None:
    blocks_path = Path(os.environ["AGENTIC_BLOCKS_FILE"])
    blocks_path.write_text(
        """## Tier 0

1. **One** — `alpha`
2. **Two** — `beta`
3. **Three** — `gamma`
"""
    )
    import importlib

    import agentic.blocks

    importlib.reload(agentic.blocks)
    pending = agentic.blocks.list_pending(limit=2)
    assert [b.slug for b in pending] == ["alpha", "beta"]


def test_verdict_serializes_round_trip(tmp_path: Path) -> None:
    from agentic.verdict import CriterionScore, Verdict

    v = Verdict(
        goal="attention_and",
        attempt="superposed_query",
        run_id="20260613T120000Z",
        architecture_fit=CriterionScore(score=4, note="solid synthetic setup"),
        faithfulness=CriterionScore(score=2, note="no causal check"),
        automated_metrics={"and_sharpness_canonical": 54.6},
        overall="borderline",
        notes="Strong on demo; missing causal evidence.",
    )
    path = v.write(tmp_path / "verdict.json")
    loaded = Verdict.load(path)
    assert loaded.goal == "attention_and"
    assert loaded.architecture_fit is not None
    assert loaded.architecture_fit.score == 4
    assert loaded.automated_metrics["and_sharpness_canonical"] == 54.6


def test_solver_retries_when_app_fails_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-booting app.py is a retry condition: the solver loops until app.py
    boots, even when main.py + the benchmark already passed."""
    import asyncio
    from types import SimpleNamespace

    from agentic import pipeline

    goal_dir = tmp_path / "experiments" / "slug"
    attempt_dir = goal_dir / "att"
    attempt_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    # Tiny budget: 3 base (QUICK) attempts, no escalation.
    monkeypatch.setattr(
        pipeline,
        "settings",
        SimpleNamespace(
            solver_retries_base=3, solver_retries_escalated=0, solver_retries_expert=0
        ),
    )
    monkeypatch.setattr(pipeline, "emit", lambda *a, **k: None)

    calls = {"completion": 0, "boot": 0}

    async def fake_completion(tier, prompt, *, system_prompt, timeout_s):  # noqa: ANN001
        calls["completion"] += 1
        return "<<file blocks>>"

    def fake_apply_blocks(text, *, root, allowed_prefixes):  # noqa: ANN001
        (attempt_dir / "main.py").write_text("print('ran')")
        (attempt_dir / "app.py").write_text("demo = object()")

    async def fake_run_subprocess(cmd, *, n_gpus, timeout):  # noqa: ANN001
        return 0, "main ran clean"

    async def fake_boot(app_path, *, n_gpus=1, wait_s=60):  # noqa: ANN001
        calls["boot"] += 1
        # First app.py is broken; the second one boots.
        if calls["boot"] < 2:
            return False, "ImportError: cannot import name 'gr.TabsItem'"
        return True, "ok"

    monkeypatch.setattr(pipeline, "_completion_with_timeout", fake_completion)
    monkeypatch.setattr(pipeline, "apply_blocks", fake_apply_blocks)
    monkeypatch.setattr(pipeline, "_run_subprocess_with_gpu", fake_run_subprocess)
    monkeypatch.setattr(pipeline, "_latest_benchmark", lambda d: {"metrics": {"score": 1.0}})
    monkeypatch.setattr(pipeline, "_boot_check_app_with_gpu", fake_boot)

    ok, metrics, _log = asyncio.run(
        pipeline._solver_with_benchmark_retries(
            "slug", "att", goal_dir, attempt_dir, gpu_requirement=1, broken_predicate=None
        )
    )

    assert ok is True
    assert metrics == {"score": 1.0}
    assert calls["boot"] == 2  # retried once after the first boot failure
    assert calls["completion"] == 2  # the boot failure triggered a fresh solver call


def test_prior_attempt_context_feeds_verdict_and_code(tmp_path: Path) -> None:
    """A retry's prompt context carries the previous attempt's verdict + source;
    a first attempt has no prior context."""
    from agentic import pipeline

    goal_dir = tmp_path / "experiments" / "attention_xor"
    first = goal_dir / "first_pass"
    first.mkdir(parents=True)
    (first / "main.py").write_text("def model_fn(b):\n    return b.q  # cheated")
    (first / "app.py").write_text("import gradio as gr\ndemo = gr.Blocks()")
    (first / "README.md").write_text("# What I did\nHardcoded the answer.")
    (first / "verdict.json").write_text('{"overall": "fail", "notes": "faked it"}')

    # First attempt: nothing to learn from yet.
    assert pipeline._prior_attempt_context("attention_xor", "first_pass", goal_dir) == ""

    # Second attempt: pulls in the prior verdict + all three source files.
    ctx = pipeline._prior_attempt_context("attention_xor", "pass_2", goal_dir)
    assert "PREVIOUS ATTEMPT (first_pass)" in ctx
    assert '"overall": "fail"' in ctx  # the jury verdict
    assert "cheated" in ctx  # prior main.py
    assert "gr.Blocks()" in ctx  # prior app.py
    assert "Hardcoded the answer." in ctx  # prior README

    # A prior folder with no main.py is skipped (nothing useful to carry).
    (goal_dir / "pass_2").mkdir()
    empty = goal_dir / "pass_3"
    empty.mkdir()
    ctx3 = pipeline._prior_attempt_context("attention_xor", "pass_3", goal_dir)
    assert "PREVIOUS ATTEMPT (first_pass)" in ctx3  # falls back past the empty pass_2
