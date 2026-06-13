"""FastAPI app serving the dashboard.

Endpoints:
    GET /              static single-page UI
    GET /api/state     full snapshot (page load / reconnect)
    GET /api/stream    SSE: pushes a fresh snapshot whenever the log changes

Stateless and read-only — it polls the two JSONL files' mtime/size and
recomputes the view on change. No DB, no websockets.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from agentic.config import settings
from agentic.dashboard.aggregate import build_view

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
