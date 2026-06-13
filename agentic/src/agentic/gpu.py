"""File-lock-backed GPU semaphore.

Each GPU has a lockfile under `<state_dir>/gpu_locks/gpu_<id>.lock`. To acquire
a slot, we open the file and call `fcntl.flock(LOCK_EX | LOCK_NB)`. The kernel
releases the lock when the holding open-file-description goes away, so a shell
that dies mid-run can't leave a stale lock — no PID dance required.

Usage:

    with acquire_gpus(n) as gpu_ids:
        subprocess.run(cmd, env={..., "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids))})

`acquire_gpus` blocks until `n` slots are free. Pair with `asyncio.to_thread`
when called from async code so the event loop stays responsive.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import time
from collections.abc import Iterator
from io import TextIOWrapper
from pathlib import Path

from agentic.config import settings


def _lock_dir() -> Path:
    p = Path(settings.state_dir) / "gpu_locks"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _try_lock(gpu_id: int) -> TextIOWrapper | None:
    """Try to take an exclusive lock on gpu_<id>.lock. Returns the open file
    (caller closes to release) or None if the slot is busy."""
    path = _lock_dir() / f"gpu_{gpu_id}.lock"
    f = path.open("w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return None
    f.write(f"{os.getpid()}\n")
    f.flush()
    return f


@contextlib.contextmanager
def acquire_gpus(n: int = 1, *, poll_s: float = 0.5) -> Iterator[list[int]]:
    """Block until `n` GPU slots are free. Yields their IDs as a list.

    Releases all held slots on context exit, even on exception. Raises
    `ValueError` if `n` exceeds the configured pool size.
    """
    if n < 1:
        yield []
        return
    if n > settings.gpu_count:
        raise ValueError(
            f"requested {n} GPUs but pool has {settings.gpu_count} slots (AGENTIC_GPU_COUNT)"
        )

    held_files: list[TextIOWrapper] = []
    held_ids: list[int] = []

    try:
        while len(held_ids) < n:
            for gpu_id in range(settings.gpu_count):
                if gpu_id in held_ids:
                    continue
                f = _try_lock(gpu_id)
                if f is not None:
                    held_files.append(f)
                    held_ids.append(gpu_id)
                    if len(held_ids) >= n:
                        break
            if len(held_ids) < n:
                time.sleep(poll_s)
        yield held_ids
    finally:
        for f in held_files:
            with contextlib.suppress(OSError):
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            f.close()
