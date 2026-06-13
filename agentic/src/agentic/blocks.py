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


_HEADING_RE = re.compile(r"^#{2,3} +.+$", re.MULTILINE)
# A task is a numbered list item whose title line carries an inline backtick
# slug, e.g. ``3. **NOT / negation** — `attention_not` ``. The slug must sit on
# the title line (`[^\n]*?` never crosses a newline), so bullet text below an
# item can mention other slugs without being mistaken for a new task.
_ITEM_RE = re.compile(
    r"^[ \t]*\d+\.[ \t]+\*\*(?P<title>.+?)\*\*[^\n]*?`(?P<slug>[A-Za-z0-9_]+)`",
    re.MULTILINE,
)


def parse_blocks(blocks_file: str | Path | None = None) -> list[Block]:
    """Parse BLOCKS.md into Block records.

    Each `N. **Title** — \\`slug\\`` list item becomes a Block; its spec is the
    indented body (I/O, "what makes it hard", "builds on") up to the next item
    or `##`/`###` heading. Numbered items without a backtick slug are skipped.
    """
    path = Path(blocks_file or settings.blocks_file)
    if not path.exists():
        return []
    text = path.read_text()

    items = list(_ITEM_RE.finditer(text))
    # Section breaks bound each item's spec: the next item or the next heading.
    breaks = sorted([m.start() for m in items] + [m.start() for m in _HEADING_RE.finditer(text)])

    blocks: list[Block] = []
    for m in items:
        title = m.group("title").strip()
        slug = m.group("slug").strip()
        line_end = text.find("\n", m.end())
        body_start = line_end + 1 if line_end != -1 else len(text)
        later = [b for b in breaks if b > m.start()]
        body_end = min(later) if later else len(text)
        spec = text[body_start:body_end].strip()
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


# Non-terminal states a crashed pipeline can leave a slug stuck in. None of
# these survive a process death — they always represent in-flight work that
# is now gone, so reconcile resets them to `pending`.
_NON_TERMINAL: tuple[BlockStatus, ...] = (
    "claimed",
    "solving",
    "pending_solver",
    "awaiting_jury",
)


def reconcile(experiments_dir: str | Path = "experiments") -> list[tuple[str, str, str]]:
    """Bring block state back in line with reality after a crash or manual edit.

    Two repairs, each appending one corrected state record:

    1. **Stuck in-flight** — a slug in a non-terminal state (the pipeline died
       mid-run) is reset to `pending` so `next_pending` / `pipeline-multi` pick
       it up again. GPU locks need no repair: they are `flock`-based and the
       kernel frees them on process death.
    2. **Dangling reference** — a slug recorded as `graded` whose `attempt`
       directory or `verdict_path` no longer exists on disk (e.g. the attempt
       was deleted by hand) is reset to `pending`.

    Returns a list of `(slug, old_status, new_status)` for everything changed.
    """
    exp = Path(experiments_dir)
    changed: list[tuple[str, str, str]] = []
    for slug, state in load_state().items():
        new_status: BlockStatus | None = None

        if state.status in _NON_TERMINAL:
            new_status = "pending"
        elif state.status == "graded":
            attempt_ok = bool(state.attempt) and (exp / slug / str(state.attempt)).is_dir()
            verdict_ok = state.verdict_path is not None and Path(state.verdict_path).is_file()
            if not (attempt_ok and verdict_ok):
                new_status = "pending"

        if new_status is not None:
            changed.append((slug, state.status, new_status))
            update_state(slug, status=new_status, attempt=None, verdict_path=None)
    return changed
