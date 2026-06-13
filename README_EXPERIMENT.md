# Running an experiment

You are an agent that was just spawned to attempt a mechanistic interpretability
experiment.

**Before anything else**: read `experiments/<your-assigned-goal>/README.md`. That
file states the specific question you are trying to answer. This document tells
you the shape your work should take — same for every goal — so you can focus on
the interpretability problem, not the scaffolding.

## Two-level layout

```
experiments/
├── pyproject.toml              # shared deps for every attempt (torch, transformers, datasets, gradio)
└── <goal>/                     # one mech-interp question
    ├── README.md               # ← goal spec; read this first
    ├── task.py                 # ← data generator + evaluator; import via load_task
    ├── benchmark.py            # ← the metric every attempt at this goal is scored on
    └── <attempt_name>/         # ← one approach to the goal; this is your workspace
        ├── pyproject.toml      # symlink to ../../pyproject.toml
        ├── README.md           # what you did + why your visualisation is right
        ├── main.py             # runs the experiment, dumps artefacts + benchmark.json
        ├── app.py              # Gradio: Demo tab (your viz) + Benchmark tab (history across attempts)
        └── results/<run-id>/   # outputs (managed by agentic.experiments.results_dir)
            ├── ...             # your custom artefacts (CSV, NPY, JSON, ...)
            └── benchmark.json  # written by agentic.experiments.record_benchmark
```

A goal has many attempts — that is the point. Pick an `attempt_name` that
describes your angle (`attention_head_ablation`, `sae_basis_swap`,
`direct_logit_attribution`), not `attempt_1`.

Your `main.py` imports the goal's `task.py` via
`agentic.experiments.load_task(__file__)` so the data and the canonical sweep
are byte-identical across every attempt at the same goal. You only ever
contribute a **model function** — the analytical/hand-built/trained piece
this attempt is testing — and pass it to `task.evaluate(...)`. The returned
payload goes straight into `record_benchmark`.

## Scaffold

From the repo root:

```bash
mkdir -p experiments/<goal>/<attempt_name>
cd experiments/<goal>/<attempt_name>
ln -s ../../pyproject.toml pyproject.toml
```

You inherit the shared workspace venv. Do not create a separate one. If you absolutely need
a dependency that is not already in `experiments/pyproject.toml`, add it there —
it is the single source of truth for every attempt.

## Model: stay close to `base_model.py`

Whatever you train should still *look like* a transformer — even loosely.
The starting point is `experiments/base_model.py`: a small stack of
self-attention + MLP + residual-stream blocks with a token embedding.
Your job is to find the **smallest delta** from that file that solves
the goal.

- **Minimal changes from `base_model.py`.** A few extra projections, a
  tweak to the softmax (extra denominator term, temperature), a
  different positional encoding, a small QKV convolution, swapping one
  layer's hyperparameters — all fine if a concrete problem motivates
  them. Rewriting the model wholesale, swapping attention for an LSTM
  or SSM, or importing a third-party architecture is out of scope.
- **Minimal layers to solve the problem.** Start with one block. Add a
  second only once you can show the first is genuinely insufficient
  (an ablation, a failed run, a circuit-level argument). Three or more
  blocks needs a stronger justification, ideally tied to the problem's
  structure.
- **Attention alone is allowed.** If a single attention layer with no
  MLP is enough to learn the target function, that is an excellent
  result — drop the MLP and say so in your README. The same applies to
  multi-layer attempts: any block where the MLP is dead weight should
  be stripped. The MLP is a tool, not a requirement.
- **Document the diff.** In your attempt's `README.md`, describe the
  model as "`base_model.py` plus *X*", not as a from-scratch design.
  The grader should be able to read your delta in a couple of lines.

This keeps attempts comparable across the same goal, and forces the
real discovery question — *what does attention (with at most a touch of
MLP) need in order to express this?* — to stay in the foreground.

## The loop

1. **Read** `experiments/<goal>/README.md`. Internalise the question and the
   model/dataset it points at.
2. **Pick a hypothesis** and name your attempt after it.
3. **Scaffold** the directory and the symlink as above.
4. **Compute** in `main.py`. Use
   `agentic.experiments.results_dir(__file__)` so artefacts land under
   `results/<utc-timestamp>/`. Save the reduced form needed for the
   visualisation; only save raw tensors when the viz genuinely needs them.
   Then hand the goal-shaped payload to
   `agentic.experiments.record_benchmark(__file__, run_dir, payload)` — that
   writes `benchmark.json` next to your other artefacts.
5. **Visualise** in `app.py`. Build a Gradio Blocks app with two tabs:
   - **Demo** — the latest-run-by-default interactive view of your result.
     Let the grader pick older runs from a dropdown.
   - **Benchmark** — drop in
     `agentic.experiments.benchmark_panel(<goal_dir>)`. It scans every
     attempt under the goal, shows a leaderboard, and plots metric history
     over runs so iteration shows up as a curve.
6. **Document** in `README.md`. Two sections, no more:
   - **What I did** — the approach in 3–6 sentences.
   - **Why this visualisation** — what about the chart/heatmap/diagram lets a
     human check the claim. Justify the axes, the comparison, what is
     emphasised.
7. **Iterate** until the viz tells the story without the README. The README is
   the safety net, not the argument.

### Boot-check `app.py`

Before declaring done, verify the Gradio app loads cleanly. The pipeline's
boot-check is structural — it imports `app.py` and asserts the module exposes
a `demo: gr.Blocks`, which catches the bugs an attempt usually has (Gradio
API misuse, import errors, missing `demo`, wrong type) without binding a port.

Two hard requirements on `app.py`:

1. Expose `demo: gr.Blocks` at the module level. The boot-check looks for
   exactly that attribute.
2. Every Gradio event/lifecycle call (`btn.click(...)`, `dd.change(...)`,
   `demo.load(...)`, etc.) must live INSIDE the `with gr.Blocks() as demo:`
   block. Calling any of them at module level raises `AttributeError: Cannot
   call X outside of a gradio.Blocks context` at import time.

The canonical shape:

```python
import gradio as gr

with gr.Blocks() as demo:
    ...
    btn.click(my_fn, inputs=..., outputs=...)
    demo.load(initial_fn, inputs=..., outputs=...)

if __name__ == "__main__":
    demo.launch()
```

To run the same check yourself:

```bash
uv run python -c "
import importlib.util, sys, gradio as gr
spec = importlib.util.spec_from_file_location('app', 'app.py')
m = importlib.util.module_from_spec(spec); sys.modules['app'] = m
spec.loader.exec_module(m)
assert isinstance(m.demo, gr.Blocks), 'app.py must expose demo: gr.Blocks'
print('ok')"
```

## Benchmarking — shared per goal

Every goal owns a `benchmark.py` that defines the metric all its attempts are
judged on. It exports:

```python
VERSION: int                              # bump if the formulae change
def score(payload: dict) -> dict[str, float | int]:
    ...
```

The goal's `README.md` documents the **payload contract** — exactly what keys
and types `score()` expects. Your `main.py` builds that payload and calls
`record_benchmark`; you never re-implement the metric. The framework writes
`benchmark.json` to your run directory and the **Benchmark** tab in any
attempt's Gradio app reads every attempt's history, so iterating on an
existing attempt shows up as a moving line and a new attempt shows up as a
new series in the leaderboard.

If you change the goal's metric, bump `VERSION` — the panel groups by version
so old runs stay legible without polluting the new series.

## How it is graded

The grader walks the rubric in priority order — an earlier item failing
dominates a later one passing. A polished visualisation cannot redeem a method
that the model doesn't actually use; a clean ablation does not need a perfect
chart to land.

### Automated metrics

The goal's `benchmark.py` is the first part of the rubric. Open the
**Benchmark** tab in any attempt's Gradio app to see latest values per
attempt × metric. Each goal's `README.md` documents which metrics matter most.
The framework expects automated metrics to cover:

- a **headline summary** value an attempt should optimise (e.g.
  `superposition_robustness` for `attention_and`);
- **robustness** across the most realistic axis of variation (concept-direction
  cosine, input noise, model scale — whatever the goal's question hinges on);
- a **baseline / strawman** measured under the same conditions.

If you want to add a checkable item to the human rubric below, ask first
whether it could live in `benchmark.py` instead. See `README_BENCHMARK.md` for
how to design a goal's benchmark.

### Human-judged criteria, in priority order

1. **Architecture fit.** Does the proposed mechanism actually solve the goal's
   task — qualitatively and quantitatively? An attempt that exhibits the
   target behaviour on a contrived example but doesn't address the goal's
   question scores low here regardless of everything else. The model is
   expected to be a small delta from `experiments/base_model.py` with the
   minimum number of layers needed; large architectural rewrites count
   against this item even when they work.
2. **Baseline comparison.** Does the attempt show an obvious strawman failing
   where the method succeeds? "X works" is one claim; "X works while
   no-`exp`/no-attention/no-circuit doesn't" is the testable one.
3. **Faithfulness / causal evidence.** Does the *model* actually use this
   mechanism? An ablation or activation-patching check that knocks out the
   proposed circuit and watches the target behaviour break is the difference
   between "a possible solution" and "the observed one". For purely synthetic
   attempts that don't run on a model, say so explicitly in your README — and
   propose what such a check would look like.
4. **Operating range.** Does the method hold up across ≥ 2 orders of magnitude
   of input scale (or the goal's relevant axis)? Degrading at the edges is
   fine if you show where it breaks; silent failure outside the demo regime is
   not.
5. **Hardcoded weights (bonus).** Can the mechanism be written out by hand
   instead of learned? A hand-set circuit that reproduces the target behaviour
   is strong evidence you understand the mechanism. Worth one extra grade tier.
6. **Visual judgement.** The grader launches your `app.py`, interacts with
   the Demo tab, and decides whether the visualisation makes your claim
   legible. A bar chart that compresses the result into one comparison is
   often better than a 12-panel heatmap.
7. **Visualisation rationale.** The grader reads the *Why this visualisation*
   section of your `README.md`. A strong entry connects the chart choice to
   what the goal asks: the right thing on the y-axis, the right baseline, the
   right grain.

**Coming later (placeholders — do not optimise yet)**

- Quantitative reproducibility across seeds and model checkpoints
- Cross-attempt code review for soundness of the interp method

## Conventions worth following

- One attempt = one hypothesis. If you find yourself testing two unrelated
  ideas, fork a second `attempt_name`.
- `main.py` runs end-to-end with no required flags. Take CLI args for tuning,
  with sensible defaults. Do not make the grader edit the file.
- `app.py` defaults to the most recent run under `results/`.
- Name and load model checkpoints explicitly in `main.py`. Do not rely on a
  global cache the grader cannot reproduce.
- When in doubt about what to visualise: pick the smallest artefact that, if
  flipped or zeroed out, would change the claim. Show that.
