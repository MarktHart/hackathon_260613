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
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

from agentic.blocks import load_state, parse_blocks
from agentic.config import settings


class UnknownSlug(ValueError):
    """Raised when a start request names a slug that isn't in BLOCKS.md."""


class UnknownAttempt(ValueError):
    """Raised when an app-launch request names an attempt that doesn't exist."""


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


def launch_slug(slug: str, *, force: bool = True, resume: bool = False) -> int:
    """Validate `slug` against BLOCKS.md and spawn a pipeline run for it.

    Raises `UnknownSlug` if the slug isn't a known task. `force` re-runs a
    slug that's already graded (the pipeline early-exits otherwise). `resume`
    skips picker/reviewer stages a prior run already passed.
    """
    if slug not in known_slugs():
        raise UnknownSlug(slug)
    cmd = [sys.executable, "-m", "agentic.cli", "pipeline", "--slug", slug]
    if force:
        cmd.append("--force")
    if resume:
        cmd.append("--resume")
    return _spawn(cmd, f"{slug}.log")


def launch_pending(count: int | None = None) -> int:
    """Spawn `pipeline-multi` to fan out across pending tasks. Returns the pid."""
    cmd = [sys.executable, "-m", "agentic.cli", "pipeline-multi"]
    if count is not None:
        cmd += ["-c", str(count)]
    return _spawn(cmd, "pipeline-multi.log")


def failed_slugs() -> list[str]:
    """Slugs whose block state is `failed` (a stalled run leaves them here)."""
    return [slug for slug, st in load_state().items() if st.status == "failed"]


# --- Gradio app launching -------------------------------------------------
#
# Attempts ship a Gradio `app.py`. The dashboard launches it on demand as a
# detached subprocess (gradio binds its own port via GRADIO_SERVER_PORT) and
# hands the URL back so the browser can open it in a new tab. Launched apps are
# remembered so a second click on the same attempt reuses the live server.

_ATTEMPT_RE = re.compile(r"^(first_pass|pass_\d+)$")
_APP_PORT_BASE = 7900
_APP_PORT_SPAN = 200

# (slug, attempt) -> {"port": int, "pid": int} for apps we've launched.
_apps: dict[tuple[str, str], dict[str, int]] = {}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _port_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _pick_port() -> int:
    """First port in the dashboard's range that nothing is already listening on."""
    taken = {a["port"] for a in _apps.values()}
    for port in range(_APP_PORT_BASE, _APP_PORT_BASE + _APP_PORT_SPAN):
        if port not in taken and not _port_listening(port):
            return port
    raise RuntimeError("no free port for the Gradio app")


def _attempt_dir(slug: str, attempt: str) -> Path:
    """Validate slug+attempt and return the attempt dir, or raise UnknownAttempt."""
    if slug not in known_slugs():
        raise UnknownSlug(slug)
    if not _ATTEMPT_RE.match(attempt):
        raise UnknownAttempt(attempt)
    d = Path("experiments") / slug / attempt
    if not (d / "app.py").is_file():
        raise UnknownAttempt(f"{slug}/{attempt} has no app.py")
    return d


def launch_app(slug: str, attempt: str, *, wait_s: int = 45) -> str:
    """Launch (or reuse) the attempt's Gradio app and return its local URL.

    Idempotent per (slug, attempt): if the app we previously launched is still
    alive and listening, its URL is returned without spawning a second server.
    Otherwise a fresh `python app.py` is spawned with GRADIO_SERVER_PORT pinned,
    and we wait up to `wait_s` for it to start accepting connections.
    """
    d = _attempt_dir(slug, attempt)
    key = (slug, attempt)

    existing = _apps.get(key)
    if existing and _pid_alive(existing["pid"]) and _port_listening(existing["port"]):
        return f"http://127.0.0.1:{existing['port']}"

    port = _pick_port()
    env = os.environ.copy()
    env["GRADIO_SERVER_PORT"] = str(port)
    env["GRADIO_SERVER_NAME"] = "127.0.0.1"
    env["PYTHONUNBUFFERED"] = "1"
    if settings.hf_home:
        env["HF_HOME"] = settings.hf_home

    log = _runs_dir() / f"app_{slug}_{attempt}.log"
    out = log.open("ab")
    proc = subprocess.Popen(  # noqa: S603 — slug/attempt are validated above
        ["uv", "run", "--project", os.getcwd(), "python", str((d / "app.py").resolve())],
        stdout=out,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        cwd=str(d),
        env=env,
    )
    _apps[key] = {"port": port, "pid": proc.pid}

    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"app for {slug}/{attempt} exited on startup (see {log.name})")
        if _port_listening(port):
            return url
        time.sleep(0.5)
    # Slow boot (heavy checkpoint load): hand back the URL anyway. The browser
    # tab will connect once the server finishes coming up.
    return url


def launch_failed() -> list[dict[str, int | str]]:
    """Re-run every failed task, one detached pipeline process per slug.

    Returns a `{slug, pid}` record for each spawned run (empty if none failed).
    A failed task isn't graded, so the run proceeds normally; we pass `force`
    anyway to be explicit that a re-run is intended. `resume` skips the
    picker/reviewer stages when a prior run already passed them, so a retry
    only redoes the solver/jury work that actually failed.
    """
    launched: list[dict[str, int | str]] = []
    for slug in failed_slugs():
        pid = launch_slug(slug, force=True, resume=True)
        launched.append({"slug": slug, "pid": pid})
    return launched
