"""Parse and apply structured file-block output from completion-mode tiers.

Tier 2 and Tier 3 are one-shot completions — they can't write files directly.
Instead they return their output in this format:

    <<FILE: experiments/foo/README.md>>
    # Title

    Content here.
    <<END FILE>>

    <<FILE: experiments/foo/benchmark.py>>
    VERSION = 1

    def score(payload):
        ...
    <<END FILE>>

The pipeline parses these blocks and writes each one to disk. The contract is
intentionally tag-based (not fences or JSON) so file contents can include any
characters except the literal closing marker `<<END FILE>>`.
"""

from __future__ import annotations

import re
from pathlib import Path

OUTPUT_CONTRACT = """\
Return your output as one or more file blocks. Each block has this exact form:

<<FILE: <relative path>>>
<file content>
<<END FILE>>

Rules:
- One block per file. Repeat the pattern for multiple files.
- Path is relative to the repo root.
- The line after `<<FILE: ...>>>` is the first line of file content.
- The line `<<END FILE>>` on its own ends the block.
- Do NOT write `<<END FILE>>` inside file content. If a literal is needed,
  use a placeholder and document it in the file.
- Output ONLY file blocks. No prose before, between, or after.
"""

_BLOCK_RE = re.compile(
    r"<<FILE:\s*(?P<path>[^>]+?)\s*>>\n(?P<content>.*?)(?:\n)?<<END FILE>>",
    re.DOTALL,
)


def parse_blocks(text: str) -> list[tuple[str, str]]:
    """Parse all `<<FILE: ...>>...<<END FILE>>` blocks. Returns [(path, content), ...]."""
    return [(m.group("path").strip(), m.group("content")) for m in _BLOCK_RE.finditer(text)]


def apply_blocks(
    text: str,
    *,
    root: str | Path = ".",
    allowed_prefixes: tuple[str, ...] | None = None,
) -> list[Path]:
    """Write every file block found in `text`. Returns the paths written.

    If `allowed_prefixes` is set, paths must start with one of them — anything
    else is silently skipped. This is the cheap guard against a completion-mode
    tier trying to write outside its scoped goal directory.
    """
    root_path = Path(root).resolve()
    written: list[Path] = []
    for rel_path, content in parse_blocks(text):
        if allowed_prefixes is not None and not any(
            rel_path.startswith(prefix) for prefix in allowed_prefixes
        ):
            continue
        # Disallow path escape via `..` or absolute paths.
        if rel_path.startswith("/") or ".." in Path(rel_path).parts:
            continue
        out = (root_path / rel_path).resolve()
        # Final containment check.
        try:
            out.relative_to(root_path)
        except ValueError:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content)
        written.append(out)
    return written
