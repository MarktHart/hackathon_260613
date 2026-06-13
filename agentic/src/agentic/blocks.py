r"""Parse BLOCKS.md (human-authored) + manage state/blocks.jsonl (machine-mutated).

BLOCKS.md is the task queue. Each block is a `## <Title>` section followed by
a `Goal: <slug>` line and free-form spec text. State (claim/solve/grade) is
appended to `state/blocks.jsonl` so BLOCKS.md stays merge-friendly.

State machine:
    pending → claimed → solving → graded
                    \-> failed
                    \-> awaiting_jury  (when --skip-jury was passed)
                    \-> pending_solver (when --skip-solver was passed)
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from agentic.config import settings

BlockStatus = Literal[
    "pending",
    "claimed",
    "pending_solver",
    "solving",
    "awaiting_jury",
    "graded",
    "failed",
]


@dataclass
class Block:
    """A task defined in BLOCKS.md."""

    slug: str
    title: str
    spec: str


@dataclass
class BlockState:
    """Current pipeline state for one slug. Mutated by appending JSONL records."""

    slug: str
    status: BlockStatus = "pending"
    attempt: str | None = None
    claimed_at: str | None = None
    updated_at: str | None = None
    verdict_path: str | None = None
    notes: list[str] = field(default_factory=list)


_HEADING_RE = re.compile(r"^## +(.+?)\s*$", re.MULTILINE)
_GOAL_RE = re.compile(r"^Goal:\s*(\S+)\s*$", re.MULTILINE)


def parse_blocks(blocks_file: str | Path | None = None) -> list[Block]:
    """Parse BLOCKS.md into Block records. Skips blocks without a `Goal:` line."""
    path = Path(blocks_file or settings.blocks_file)
    if not path.exists():
        return []
    text = path.read_text()

    # Find every `## Title` heading and its body up to the next `## ` or EOF.
    blocks: list[Block] = []
    matches = list(_HEADING_RE.finditer(text))
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]

        slug_match = _GOAL_RE.search(body)
        if not slug_match:
            continue
        slug = slug_match.group(1).strip()
        spec = _GOAL_RE.sub("", body).strip()
        blocks.append(Block(slug=slug, title=title, spec=spec))
    return blocks


def _state_path() -> Path:
    p = Path(settings.state_dir) / "blocks.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_state() -> dict[str, BlockState]:
    """Latest state per slug, last-write-wins over the JSONL log."""
    path = _state_path()
    if not path.exists():
        return {}
    states: dict[str, BlockState] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        slug = data.get("slug")
        if not isinstance(slug, str):
            continue
        states[slug] = BlockState(
            slug=slug,
            status=data.get("status", "pending"),
            attempt=data.get("attempt"),
            claimed_at=data.get("claimed_at"),
            updated_at=data.get("updated_at"),
            verdict_path=data.get("verdict_path"),
            notes=list(data.get("notes") or []),
        )
    return states


def update_state(slug: str, **changes: Any) -> BlockState:
    """Apply `changes` to the current state of `slug` and append a new record."""
    states = load_state()
    state = states.get(slug, BlockState(slug=slug))
    for k, v in changes.items():
        setattr(state, k, v)
    state.updated_at = datetime.now(UTC).isoformat()

    path = _state_path()
    with path.open("a") as f:
        f.write(json.dumps(asdict(state)) + "\n")
    return state


def next_pending(blocks_file: str | Path | None = None) -> Block | None:
    """First block in BLOCKS.md order whose current state is pending."""
    blocks = parse_blocks(blocks_file)
    states = load_state()
    for b in blocks:
        s = states.get(b.slug)
        if s is None or s.status == "pending":
            return b
    return None


def list_pending(limit: int | None = None, blocks_file: str | Path | None = None) -> list[Block]:
    """All pending blocks in BLOCKS.md order, up to `limit` if given."""
    blocks = parse_blocks(blocks_file)
    states = load_state()
    out: list[Block] = []
    for b in blocks:
        s = states.get(b.slug)
        if s is None or s.status == "pending":
            out.append(b)
            if limit is not None and len(out) >= limit:
                break
    return out
