#!/usr/bin/env bash
#
# kill.sh — stop all detached pipeline runs, then GC the block state.
#
# The dashboard launches each task as a detached `python -m agentic.cli
# pipeline --slug ... --force` process (reparented to PID 1). Those drive the
# claude SDK agents and GPU subprocesses. Killing them mid-run leaves slugs
# stuck in non-terminal states (claimed/solving/...), so we follow up with
# `agentic reconcile` to reset them to `pending` and bring state back in line.
#
# Deliberately NOT touched: the dashboard, and any interactive `claude`
# session (killing by the `claude` name would take this session down too —
# the SDK-spawned agents die as children of the pipeline pythons we kill here).

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

PIPELINE_PAT='agentic.cli pipeline'
SMOKE_PAT='agentic_smoke_'

echo "==> Killing detached pipeline runs ('$PIPELINE_PAT') ..."
pkill -f "$PIPELINE_PAT" 2>/dev/null && echo "    sent SIGTERM" || echo "    none running"

# Give them a moment to exit (and let their claude/GPU children unwind), then
# escalate to SIGKILL for any survivors.
for _ in 1 2 3 4 5; do
  pgrep -f "$PIPELINE_PAT" >/dev/null 2>&1 || break
  sleep 1
done
if pgrep -f "$PIPELINE_PAT" >/dev/null 2>&1; then
  echo "==> Survivors after SIGTERM — escalating to SIGKILL ..."
  pkill -9 -f "$PIPELINE_PAT" 2>/dev/null
  sleep 1
fi

# Sweep any orphaned smoke-test subprocesses left behind mid-stage.
if pgrep -f "$SMOKE_PAT" >/dev/null 2>&1; then
  echo "==> Sweeping orphaned smoke subprocesses ('$SMOKE_PAT') ..."
  pkill -9 -f "$SMOKE_PAT" 2>/dev/null
fi

# Pick an interpreter that matches how the runs were launched.
if [ -x "$REPO/.venv/bin/python3" ]; then
  RUN=("$REPO/.venv/bin/python3" -m agentic.cli)
elif command -v uv >/dev/null 2>&1; then
  RUN=(uv --project agentic run agentic)
else
  RUN=(python3 -m agentic.cli)
fi

echo "==> Running GC (agentic reconcile) to restore valid state ..."
"${RUN[@]}" reconcile

# Report what's still alive so the result is verifiable.
echo "==> Remaining pipeline processes:"
if pgrep -af "$PIPELINE_PAT" 2>/dev/null; then
  echo "    WARNING: some pipeline processes are still alive (see above)"
else
  echo "    none — all stopped"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> GPU processes (check for leftover VRAM holders):"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
    || echo "    (nvidia-smi query unavailable)"
fi

echo "==> Done."
