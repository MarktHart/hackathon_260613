# mech-interp pipeline

## Why this exists

Software pipelines verify themselves two ways. **Unit tests** pin down strict,
known-answer functionality, and increasingly **LLM-as-a-judge** adds a softer
layer of supervision for the things that are hard to assert mechanically. But
an LLM judge has a hard time certifying anything beyond its own capabilities —
it can only reliably grade what it could already work out itself, so the
genuinely hard cases slip straight past it.

ML projects can't lean on the first layer. The whole reason you reach for ML is
that you *don't* know how to write the function — so you can't write the unit
test that checks it. That leaves machine-learning work without strict
verification.

The usual substitute is a **benchmark**. But benchmarks on real-world use cases
are extremely expensive to build: labelled data, careful task design, and a
metric everyone trusts.

This repo's bet: for a large class of problems you can **generate useful
synthetic benchmarks automatically** by leaning on old-school,
exactly-known algorithms as ground truth — and then **hill-climb against that
much cleaner signal**. The domain here is mechanistic interpretability: the
task queue in `BLOCKS.md` is largely built from classical algorithms (AND,
argmax, histogram, prefix-sum, …), each asking whether a small transformer can
implement that behaviour — and the classical algorithm gives the exact answer
to grade against.

Seen this way, an auto-generated synthetic benchmark is a **middle ground**
between the two layers: cheaper and broader than hand-written unit tests, yet
anchored to exact ground truth the LLM judge can't supply on its own. Nothing
about this is ML-specific — an ordinary (non-ML) project could lean on the same
trick to cover the gap between its unit tests and its LLM judge.

Each attempt is then scored on three independent signals:

1. **The hard benchmark score** — automated metrics from `benchmark.py`: exact,
   cheap, repeatable ground truth.
2. **An LLM jury** — grades the *quality of the solution and its evidence* on a
   rubric the benchmark can't capture: architecture fit, a baseline/strawman
   comparison, faithfulness (causal evidence the model actually *uses* the
   mechanism), operating range, and visualisation quality, among others.
3. **The human** — the final judge, convinced (or not) through each attempt's
   Gradio dashboard.

Climbing the hill is just re-running. Each new attempt gets its own folder
under `experiments/<goal>/`, reads the prior attempt's verdict and code, and is
told to diagnose what fell short and take a meaningfully different approach;
retries within a run also escalate to stronger model tiers.

## How it works

Each task ("goal") flows through four stages. Different stages run on different
model tiers (cheap → expensive) to keep cost down; see [Tier mapping](#tier-mapping).

| Stage | What it does | Produces |
|-------|--------------|----------|
| **picker** | claims the next pending block from `BLOCKS.md` | the goal's `benchmark.py` + `README.md` |
| **reviewer** | checks and, if needed, edits the benchmark for correctness | a reviewed `benchmark.py` |
| **solver** | writes an attempt at the goal; the pipeline then runs it | `main.py`, `app.py`, `benchmark.json` |
| **jury** | grades the attempt against the benchmark | `verdict.json` |

Everything lands under `experiments/<goal>/<attempt>/`. State and an
append-only event log live under `state/` (gitignored).

## Setup

```bash
# Install deps into the shared workspace venv.
uv sync

# Configure keys + tier endpoints.
cp .env.example .env
# then fill in at least ANTHROPIC_API_KEY and NEBIUS_API_KEY
```

Tier defaults: tier 1 (reviewer + jury) → Anthropic; tiers 2 & 3 (picker +
solver) → Nebius Token Factory. Everything else is overridable via env — see
`.env.example`.

## Quick start

```bash
# Option A — drive it from the live web dashboard (recommended).
uv run agentic dashboard
# → http://127.0.0.1:8080  (start/retry runs from the UI, watch progress live)

# Option B — run one task straight from the CLI.
uv run agentic pipeline            # picks the next pending block
```

## Pipeline commands

```bash
# Pick the next pending block from BLOCKS.md and run the full pipeline.
uv run agentic pipeline

# Run on a specific goal slug.
uv run agentic pipeline -s direct_logit_attribution

# Re-grade an already-graded goal (otherwise idempotency skips it).
uv run agentic pipeline -s attention_and --force

# Reuse the picker/reviewer output a prior run already passed; re-run solver onward.
uv run agentic pipeline -s attention_and --resume

# Skip cheaper solver rungs so a re-run goes straight to a better model.
uv run agentic pipeline -s attention_and --min-tier expert   # quick|standard|expert

# Stop after the benchmark is written and reviewed (no solver).
uv run agentic pipeline -s attention_and --skip-solver

# Stop after the solver produces an attempt (no jury).
uv run agentic pipeline -s attention_and --skip-jury

# Fan out across every pending block. The 2-slot GPU pool throttles
# the subprocess stages automatically.
uv run agentic pipeline-multi

# Cap concurrent LLM stages at 3 and only process the next 5 pending blocks.
uv run agentic pipeline-multi -n 3 -c 5
```

## Other commands

```bash
# Serve the live web dashboard (tails the event log; start/retry runs from the UI).
uv run agentic dashboard            # add --port/-p or --host to override

# Tail the pipeline event log.
uv run agentic events -n 50

# Repair block state to match what's on disk (after a crash). See `--help` for what it fixes.
uv run agentic reconcile
```

## Direct experiment commands

Run a single attempt's code without the pipeline:

```bash
# Compute + write benchmark.json for one attempt.
uv run python experiments/attention_and/superposed_query/main.py

# Launch that attempt's Gradio dashboard (Demo tab + Benchmark tab).
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
| `agentic/` | the framework — pipeline, tier runner, gpu pool, events, blocks parser, web dashboard, usage tracker |
| `experiments/<goal>/` | one mech-interp question — `README.md` + `benchmark.py` + attempt subdirs |
| `experiments/<goal>/<attempt>/` | one attempt — `main.py`, `app.py`, `README.md`, `results/<run-id>/`, `verdict.json` |
| `state/blocks.jsonl` | per-slug pipeline state (gitignored) |
| `state/events.jsonl` | append-only event log (gitignored) |

## Tier mapping

| Stage | Tier | Default model | Notes |
|-------|------|---------------|-------|
| picker | STANDARD (#2) | `nvidia/Nemotron-3-Ultra-550b-a55b` | one-shot; writes goal README + benchmark.py |
| reviewer | EXPERT (#1) | `claude-opus-4-8` (effort=high) | full agent loop; may edit |
| solver | QUICK (#3) | `nvidia/Cosmos3-Super-Reasoner` | one-shot; pipeline runs the code |
| jury | EXPERT (#1) | `claude-opus-4-8` (effort=high) | full agent loop; writes verdict.json |

Stages escalate tiers on retry (picker: STANDARD→EXPERT; solver:
QUICK→STANDARD→EXPERT). GPU slots, wall-clock budgets per tier, retry counts,
the `is_obviously_broken` short-circuit, and the shared `HF_HOME` are all
configurable via env — see `.env.example`.
