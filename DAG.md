# Pipeline DAG

Stages, inputs, outputs, and side effects of `agentic pipeline` (one slug)
and `agentic pipeline-multi` (fan-out). Every stage emits an event to
`state/events.jsonl` and updates per-slug status in `state/blocks.jsonl`.

## One-slug pipeline

```
                  BLOCKS.md
                      │
                      ▼
              ┌───────────────┐
              │  next_pending │  (or explicit --slug)
              │   blocks.py   │
              └───────┬───────┘
                      │  slug, title, spec
                      ▼
        ┌──────────────────────────────────────────┐
        │  PICKER ↻ SMOKE-TEST LOOP                │
        │                                          │
        │  ┌────────────────────┐                  │
        │  │  PICKER            │ STANDARD (cmpl)  │
        │  │  + smoke feedback  │   then           │
        │  │                    │ EXPERT (agentic) │
        │  └─────────┬──────────┘                  │
        │            │ writes README/task/bench    │
        │            ▼                             │
        │  ┌────────────────────┐                  │
        │  │  SMOKE TEST        │ CPU-only, ≤60s   │
        │  │  task.evaluate(    │                  │
        │  │   random_model_fn  │                  │
        │  │  ) → score()       │                  │
        │  └─────────┬──────────┘                  │
        │     fail   │   pass                      │
        │     ◄──────┴──────►                      │
        │  feed traceback                          │
        │   back into next                         │
        │   picker prompt                          │
        │                                          │
        │  budget: picker_retries_base at STANDARD │
        │        + picker_retries_escalated EXPERT │
        │  exhausted → status: failed              │
        └──────────────┬───────────────────────────┘
                       │  experiments/<slug>/{README.md, task.py, benchmark.py}
                       ▼
        ┌─────────────────────────────┐
        │  REVIEWER   (Tier 1 EXPERT) │  agentic, Claude Opus
        │  Read/Write/Edit/Glob/Grep  │  starts from a smoke-tested goal,
        │                             │  focuses on substance not shapes
        └─────────────┬───────────────┘
                      │  writes  experiments/<slug>/.review.txt
                      ▼
              ┌───────────────┐
              │ --skip-solver │── yes ──▶ status: pending_solver  ──▶ EXIT
              └───────┬───────┘
                      │ no
                      ▼
        ┌──────────────────────────────────────────┐
        │  SOLVER ↻ BENCHMARK LOOP                 │
        │                                          │
        │  ┌────────────────────┐                  │
        │  │  SOLVER            │ QUICK (cmpl)     │
        │  │  + last-run        │   then           │
        │  │    feedback        │ STANDARD (cmpl)  │
        │  └─────────┬──────────┘                  │
        │            │ writes main.py/app.py/README│
        │            ▼                             │
        │  ┌────────────────────┐                  │
        │  │  RUN main.py       │ GPU slot held    │
        │  │  → benchmark.json  │ (≤600s)          │
        │  └─────────┬──────────┘                  │
        │            │                             │
        │            ▼                             │
        │  ┌────────────────────┐                  │
        │  │  is_obviously_     │ optional         │
        │  │  broken(metrics)?  │                  │
        │  └─────────┬──────────┘                  │
        │     fail   │   pass                      │
        │  ◄─────────┴────────►                    │
        │  feed traceback or                       │
        │   degenerate metrics                     │
        │   into next prompt                       │
        │                                          │
        │  budget: solver_retries_base at QUICK    │
        │        + solver_retries_escalated  STD   │
        │  exhausted → status: failed              │
        │              (or short_circuit)          │
        └──────────────┬───────────────────────────┘
                       │  benchmark.json (passing or last)
                       ▼
        ┌─────────────────────────────┐
        │  BOOT-CHECK app.py          │  subprocess + acquire_gpus()
        │  import + assert demo:Blocks│  no port binding
        └─────────────┬───────────────┘
                      ▼
              ┌───────────────┐
              │  --skip-jury  │── yes ──▶ status: awaiting_jury  ──▶ EXIT
              └───────┬───────┘
                      │ no
                      ▼
        ┌─────────────────────────────┐
        │  JURY       (Tier 1 EXPERT) │  agentic, Claude Opus
        │  Read/Write/Glob/Grep       │  reads benchmark.json (already non-broken)
        └─────────────┬───────────────┘
                      │  writes  experiments/<slug>/<attempt>/verdict.json
                      ▼
              status: graded
```

## Stage → tier → model → mode

| Stage    | Base tier        | Escalated tier   | Default models                        | Tools |
|----------|------------------|------------------|---------------------------------------|-------|
| picker   | 2 — STANDARD     | 1 — EXPERT       | Nemotron → Claude Opus                | completion → Read/Write/Edit |
| reviewer | 1 — EXPERT       | —                | Claude Opus (effort=high)             | Read/Write/Edit/Glob/Grep |
| solver   | 3 — QUICK        | 2 — STANDARD     | Cosmos3 → Nemotron                    | completion (file blocks) |
| jury     | 1 — EXPERT       | —                | Claude Opus (effort=high)             | Read/Write/Glob/Grep |

Retry budgets and the smoke-test timeout are env-tunable:

| Setting                        | Env var                              | Default |
|--------------------------------|--------------------------------------|---------|
| `picker_retries_base`          | `AGENTIC_PICKER_RETRIES_BASE`        | 2       |
| `picker_retries_escalated`     | `AGENTIC_PICKER_RETRIES_ESCALATED`   | 1       |
| `solver_retries_base`          | `AGENTIC_SOLVER_RETRIES_BASE`        | 2       |
| `solver_retries_escalated`     | `AGENTIC_SOLVER_RETRIES_ESCALATED`   | 1       |
| `smoke_test_timeout_s`         | `AGENTIC_SMOKE_TIMEOUT_S`            | 60      |

## Smoke-test contract (added to task.py)

The picker now MUST emit `random_model_fn() -> ModelFn` in `task.py`: a
callable with the goal's real `ModelFn` signature whose body returns
random / zero values. Pure NumPy. Between picker and reviewer the
pipeline runs, in a CPU-only subprocess:

```python
payload  = task.evaluate(task.random_model_fn())
metrics  = benchmark.score(payload)
```

Any traceback is fed back into the next picker prompt as feedback. This
catches shape mismatches, payload-contract drift, and missing keys before
the agentic reviewer turn — and before any GPU-running solver attempt.

## State + control edges

```
   BLOCKS.md ──► blocks.py.next_pending ──► state/blocks.jsonl
                                            (pending → claimed → solving →
                                             graded | failed | pending_solver |
                                             awaiting_jury)

   every stage ──► events.emit ──► state/events.jsonl
       task_claimed, picker_attempt, benchmark_written,
       benchmark_smoke (ok/log_tail), benchmark_reviewed,
       attempt_started, solver_attempt, solver_main_run,
       short_circuit, attempt_done, solver_app_boot, graded,
       stage_timeout, pipeline_paused, pipeline_failed, pipeline_idle

   gpu.acquire_gpus ──► state/gpu_locks (filesystem semaphore,
                        AGENTIC_GPU_COUNT=2). Held by main.py runs +
                        app.py boot-check. NOT held by the smoke test.
```

Idempotency: `run_pipeline(slug=X)` exits early with status `skipped` if
`state/blocks.jsonl` already shows `graded` for X, unless `--force`.

## `pipeline-multi` fan-out

```
   list_pending(count) ──► [slug_1, slug_2, …, slug_n]
                                │
                                ▼  asyncio.gather, Semaphore(n_concurrent?)
         ┌──────────┬──────────┬──────────┐
         ▼          ▼          ▼          ▼
       slug_1    slug_2    slug_3      slug_n     (each runs the one-slug DAG above)
         │          │          │          │
         └──────────┴────┬─────┴──────────┘
                         ▼
              GPU semaphore (size = AGENTIC_GPU_COUNT)
              throttles all main.py + app.py subprocesses
```

LLM calls run concurrently up to `n_concurrent`; subprocess stages queue
behind the GPU pool regardless. Smoke tests run CPU-only and don't queue
on the GPU pool.

## Free-form orchestrator (separate path)

`agentic run "<task>"` and `agentic run-goal <dir>` bypass the pipeline
entirely:

```
   user prompt ──► runner.run_task
                   Claude Agent SDK loop (orchestrator_model = claude-opus-4-7)
                   tools: Read/Write/Edit/Bash/Glob/Grep/WebSearch/WebFetch
                          + Agent (researcher | experimenter | external-dispatcher)
                          + mcp__external-llm__dispatch_to_external_llm
```

The `external-dispatcher` sub-agent calls LiteLLM (`AGENTIC_EXTERNAL_MODEL`,
default `openai/gpt-4.1`) for non-Claude providers.
