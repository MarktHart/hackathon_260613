# mech-interp pipeline

A small agentic framework for running mechanistic-interpretability experiments
end-to-end. Tasks live in `BLOCKS.md`; a three-tier pipeline (picker →
reviewer → solver → jury) claims them, writes the benchmark, builds an
attempt, runs it, and grades it. Results land under `experiments/<goal>/<attempt>/`
and surface in each attempt's Gradio dashboard.

## Setup

```bash
# Install deps into the shared workspace venv.
uv sync

# Configure keys + tier endpoints.
cp .env.example .env
# then fill in at least ANTHROPIC_API_KEY and NEBIUS_API_KEY
```

Tier defaults: tier 1 (review + judge) → Anthropic; tiers 2 & 3 (benchmark +
solve) → Nebius Token Factory. All overridable via env — see `.env.example`.

## Pipeline commands

```bash
# Pick the next pending block from BLOCKS.md and run the full pipeline.
uv run agentic pipeline

# Run on a specific goal slug.
uv run agentic pipeline -s direct_logit_attribution

# Re-grade an already-graded goal (idempotency would otherwise skip it).
uv run agentic pipeline -s attention_and --force

# Stop after the benchmark is written and reviewed.
uv run agentic pipeline -s attention_and --skip-solver

# Stop after the solver produces an attempt (no jury).
uv run agentic pipeline -s attention_and --skip-jury

# Fan out across every pending block. The 2-slot GPU pool throttles
# the subprocess stages automatically.
uv run agentic pipeline-multi

# Cap concurrent LLM stages at 3 and only process the next 5 pending blocks.
uv run agentic pipeline-multi -n 3 -c 5

# Tail the event log.
uv run agentic events -n 50
```

## Direct experiment commands

```bash
# Compute + write benchmark.json for one attempt.
uv run python experiments/attention_and/superposed_query/main.py

# Launch the Gradio dashboard (Demo tab + Benchmark tab).
uv run python experiments/attention_and/superposed_query/app.py
# → http://127.0.0.1:7860
```

## Verification

```bash
# Lint + typecheck + test every project (agentic + every experiments/<goal>/<attempt>/).
bash verify.sh
```

## Where things live

| Path | What |
|------|------|
| `BLOCKS.md` | task queue — append `## Title` + `Goal: slug` blocks |
| `README_EXPERIMENT.md` | the contract a spawned solver/experimenter follows (layout, loop, grading rubric) |
| `README_BENCHMARK.md` | how goal authors design `benchmark.py` (payload contract, metrics, hooks) |
| `agentic/` | the framework — pipeline, tier runner, gpu pool, events, blocks parser, usage tracker |
| `experiments/<goal>/` | one mech-interp question — `README.md` + `benchmark.py` + attempt subdirs |
| `experiments/<goal>/<attempt>/` | one attempt — `main.py`, `app.py`, `README.md`, `results/<run-id>/` |
| `state/blocks.jsonl` | per-slug pipeline state (gitignored) |
| `state/events.jsonl` | append-only event log (gitignored) |

## Tier mapping

| Stage | Tier | Default model | Notes |
|-------|------|---------------|-------|
| picker | STANDARD (#2) | `nvidia/Nemotron-3-Ultra-550b-a55b` | one-shot; writes goal README + benchmark.py |
| reviewer | EXPERT (#1) | `claude-opus-4-8` (effort=high) | full agent loop; may edit |
| solver | QUICK (#3) | `nvidia/Cosmos3-Super-Reasoner` | one-shot; pipeline runs the code |
| jury | EXPERT (#1) | `claude-opus-4-8` (effort=high) | full agent loop; writes verdict.json |

GPU slots, wall-clock budgets per tier, retry counts with tier escalation, the
`is_obviously_broken` short-circuit, and the shared `HF_HOME` are all
configurable via env — see `.env.example`.
