#!/bin/bash
set -euxo pipefail

# One shared venv at the workspace root.
ROOT="$(pwd)"
uv sync

PROJECTS=(
    agentic
)

shopt -s nullglob
for d in experiments/*/*/; do
    # Each experiment dir is expected to symlink to experiments/pyproject.toml.
    [ -e "$d/pyproject.toml" ] && PROJECTS+=("${d%/}")
done
shopt -u nullglob

for proj in "${PROJECTS[@]}"; do
    pushd "$proj" > /dev/null
    # --project pins uv at the workspace root so tools run from the shared venv,
    # regardless of how the local pyproject.toml is wired (real file or symlink).
    uv run --project "$ROOT" ruff check --fix .
    uv run --project "$ROOT" ruff format .
    uv run --project "$ROOT" mypy .
    uv run --project "$ROOT" pytest . || [ $? -eq 5 ] # don't fail if no tests were found
    popd > /dev/null
done
