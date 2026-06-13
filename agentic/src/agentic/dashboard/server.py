"""FastAPI app serving the dashboard.

Endpoints:
    GET  /                  static single-page UI
    GET  /api/state         full snapshot (page load / reconnect)
    GET  /api/stream        SSE: pushes a fresh snapshot whenever the log changes
    POST /api/start         launch one task (spawns a detached pipeline run)
    POST /api/start-pending fan out a pipeline-multi run across pending tasks
    POST /api/retry-failed  re-run every task currently in the failed state
    POST /api/launch-app    launch an attempt's Gradio app, return its local URL

The GET side polls the two JSONL files' mtime/size and recomputes the view on
change. The POST side spawns the normal CLI as a detached subprocess (see
`launcher`); it never runs the pipeline in-process. No DB, no websockets.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agentic.config import settings
from agentic.dashboard.aggregate import build_view
from agentic.dashboard.launcher import (
    UnknownAttempt,
    UnknownSlug,
    launch_app,
    launch_failed,
    launch_pending,
    launch_slug,
)

_STATIC = Path(__file__).parent / "static"

app = FastAPI(title="agentic dashboard", docs_url=None, redoc_url=None)

# Set by serve() when a shutdown signal arrives. The SSE generator watches this
# so it can end itself promptly instead of being force-cancelled past the
# graceful-shutdown timeout (which would log a noisy cancellation traceback).
_shutdown = asyncio.Event()


def _state_files() -> list[Path]:
    d = Path(settings.state_dir)
    return [d / "events.jsonl", d / "blocks.jsonl"]


def _fingerprint() -> tuple[Any, ...]:
    """Cheap change-detector: (size, mtime) of each watched file."""
    fp: list[Any] = []
    for f in _state_files():
        try:
            st = f.stat()
            fp.append((st.st_size, st.st_mtime))
        except FileNotFoundError:
            fp.append(None)
    return tuple(fp)


@app.get("/api/state")
def api_state() -> dict[str, Any]:
    return build_view()


class StartRequest(BaseModel):
    slug: str


class StartPendingRequest(BaseModel):
    count: int | None = None


@app.post("/api/start")
def api_start(req: StartRequest) -> dict[str, Any]:
    """Launch a pipeline run for one task. Rejects unknown or already-running slugs."""
    view = build_view()
    match = next((p for p in view["problems"] if p["slug"] == req.slug), None)
    if match is not None and match["is_running"]:
        raise HTTPException(status_code=409, detail=f"{req.slug} is already running")
    try:
        pid = launch_slug(req.slug)
    except UnknownSlug:
        raise HTTPException(status_code=400, detail=f"unknown task: {req.slug}") from None
    return {"ok": True, "slug": req.slug, "pid": pid}


@app.post("/api/start-pending")
def api_start_pending(req: StartPendingRequest) -> dict[str, Any]:
    """Fan out a `pipeline-multi` run across (up to `count`) pending tasks."""
    pid = launch_pending(req.count)
    return {"ok": True, "pid": pid, "count": req.count}


class LaunchAppRequest(BaseModel):
    slug: str
    attempt: str


@app.post("/api/launch-app")
def api_launch_app(req: LaunchAppRequest) -> dict[str, Any]:
    """Launch (or reuse) an attempt's Gradio app; return its local URL."""
    try:
        url = launch_app(req.slug, req.attempt)
    except (UnknownSlug, UnknownAttempt) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None
    return {"ok": True, "slug": req.slug, "attempt": req.attempt, "url": url}


@app.post("/api/retry-failed")
def api_retry_failed() -> dict[str, Any]:
    """Re-run every task currently in the `failed` state."""
    launched = launch_failed()
    return {"ok": True, "launched": launched, "count": len(launched)}


@app.get("/api/stream")
async def api_stream() -> StreamingResponse:
    async def gen() -> AsyncIterator[str]:
        last_fp: tuple[Any, ...] | None = None
        ticks_since_push = 0
        try:
            while not _shutdown.is_set():
                fp = _fingerprint()
                # Push on change, and at least every ~10s so liveness/age refresh.
                if fp != last_fp or ticks_since_push >= 10:
                    last_fp = fp
                    ticks_since_push = 0
                    payload = json.dumps(build_view())
                    yield f"data: {payload}\n\n"
                else:
                    ticks_since_push += 1
                    yield ": keep-alive\n\n"
                # Tick once a second, but wake immediately on shutdown so the
                # stream ends itself rather than being force-cancelled.
                try:
                    await asyncio.wait_for(_shutdown.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            # Raised if the client disconnects or the stream is cancelled. Exit
            # quietly instead of surfacing an "Exception in ASGI application".
            return

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    import asyncio
    import contextlib
    import signal

    import uvicorn

    # The /api/stream SSE endpoint is an infinite generator (open as long as a
    # dashboard tab is). Without a graceful-shutdown timeout, a single Ctrl+C
    # leaves uvicorn waiting forever for that stream to finish, so the process
    # appears to hang. Capping the timeout lets uvicorn force-close lingering
    # streams and exit promptly on the first Ctrl+C.
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        timeout_graceful_shutdown=2,
    )
    server = uvicorn.Server(config)

    # Own the signal handling instead of letting uvicorn.run()/asyncio.run()
    # do it. On Python 3.12 those install competing SIGINT handlers, so Ctrl+C
    # races and dumps a KeyboardInterrupt traceback mid-shutdown. Instead we
    # run on a loop we own, neuter uvicorn's own signal capture, and add a
    # handler that just flips `should_exit` — uvicorn's main loop notices it
    # and shuts down gracefully, then serve() returns cleanly with no traceback.
    server.capture_signals = contextlib.nullcontext  # type: ignore[method-assign]
    loop = asyncio.new_event_loop()

    def _request_exit() -> None:
        server.should_exit = True
        _shutdown.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_exit)
    try:
        loop.run_until_complete(server.serve())
    finally:
        loop.close()
