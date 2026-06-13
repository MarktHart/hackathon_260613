"""Start pipeline runs from the dashboard by spawning detached subprocesses.

The dashboard process never runs the pipeline in-process (heavy GPU work that
must outlive an HTTP request and a dashboard restart). Instead it spawns the
same CLI you'd run by hand — `agentic pipeline --slug <slug>` — in a new
session, writing to the same `state/` files the dashboard already tails.

Slugs are validated against BLOCKS.md before they ever reach the argv, so a
request can't inject an arbitrary command.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from agentic.blocks import parse_blocks
from agentic.config import settings


class UnknownSlug(ValueError):
    """Raised when a start request names a slug that isn't in BLOCKS.md."""


def known_slugs() -> set[str]:
    return {b.slug for b in parse_blocks()}


def _runs_dir() -> Path:
    d = Path(settings.state_dir) / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _spawn(cmd: list[str], log_name: str) -> int:
    """Spawn `cmd` detached, logging to state/runs/<log_name>. Returns the pid."""
    log = _runs_dir() / log_name
    # Append-mode: keep prior run logs for the same slug for debugging.
    out = log.open("ab")
    proc = subprocess.Popen(  # noqa: S603 — cmd is built from validated slugs only
        cmd,
        stdout=out,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        cwd=os.getcwd(),
        env=os.environ.copy(),
    )
    return proc.pid


def launch_slug(slug: str, *, force: bool = True) -> int:
    """Validate `slug` against BLOCKS.md and spawn a pipeline run for it.

    Raises `UnknownSlug` if the slug isn't a known task. `force` re-runs a
    slug that's already graded (the pipeline early-exits otherwise).
    """
    if slug not in known_slugs():
        raise UnknownSlug(slug)
    cmd = [sys.executable, "-m", "agentic.cli", "pipeline", "--slug", slug]
    if force:
        cmd.append("--force")
    return _spawn(cmd, f"{slug}.log")


def launch_pending(count: int | None = None) -> int:
    """Spawn `pipeline-multi` to fan out across pending tasks. Returns the pid."""
    cmd = [sys.executable, "-m", "agentic.cli", "pipeline-multi"]
    if count is not None:
        cmd += ["-c", str(count)]
    return _spawn(cmd, "pipeline-multi.log")
