"""FastAPI app serving the dashboard.

Endpoints:
    GET  /                  static single-page UI
    GET  /api/state         full snapshot (page load / reconnect)
    GET  /api/stream        SSE: pushes a fresh snapshot whenever the log changes
    POST /api/start         launch one task (spawns a detached pipeline run)
    POST /api/start-pending fan out a pipeline-multi run across pending tasks

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
from agentic.dashboard.launcher import UnknownSlug, launch_pending, launch_slug

_STATIC = Path(__file__).parent / "static"

app = FastAPI(title="agentic dashboard", docs_url=None, redoc_url=None)


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
    """Fan out a `pipeline-multi` run across pending tasks."""
    pid = launch_pending(req.count)
    return {"ok": True, "pid": pid}


@app.get("/api/stream")
async def api_stream() -> StreamingResponse:
    async def gen() -> AsyncIterator[str]:
        last_fp: tuple[Any, ...] | None = None
        ticks_since_push = 0
        while True:
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
            await asyncio.sleep(1.0)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")
