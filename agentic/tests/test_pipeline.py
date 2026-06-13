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
        """# header

## First task

Goal: first_one

Spec for first.

## Second task

Goal: second_one

Spec for second.
"""
    )
    import importlib

    import agentic.blocks

    importlib.reload(agentic.blocks)

    blocks = agentic.blocks.parse_blocks()
    assert [b.slug for b in blocks] == ["first_one", "second_one"]
    assert blocks[0].title == "First task"

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
        """## One

Goal: alpha

## Two

Goal: beta

## Three

Goal: gamma
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
