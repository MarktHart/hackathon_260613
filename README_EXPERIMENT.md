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
    └── <attempt_name>/         # ← one approach to the goal; this is your workspace
        ├── pyproject.toml      # symlink to ../../pyproject.toml
        ├── README.md           # what you did + why your visualisation is right
        ├── main.py             # runs the experiment, dumps artefacts
        ├── app.py              # Gradio interface for the human grader
        └── results/<run-id>/   # outputs (managed by agentic.experiments.results_dir)
```

A goal has many attempts — that is the point. Pick an `attempt_name` that
describes your angle (`attention_head_ablation`, `sae_basis_swap`,
`direct_logit_attribution`), not `attempt_1`.

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

## The loop

1. **Read** `experiments/<goal>/README.md`. Internalise the question and the
   model/dataset it points at.
2. **Pick a hypothesis** and name your attempt after it.
3. **Scaffold** the directory and the symlink as above.
4. **Compute** in `main.py`. Use
   `agentic.experiments.results_dir(__file__)` so artefacts land under
   `results/<utc-timestamp>/`. Save the reduced form needed for the
   visualisation; only save raw tensors when the viz genuinely needs them.
5. **Visualise** in `app.py`. Build a Gradio Blocks app that loads the latest
   run by default and lets the grader pick older runs from a dropdown. Show the
   answer in the form the grader can verify (or falsify) at a glance.
6. **Document** in `README.md`. Two sections, no more:
   - **What I did** — the approach in 3–6 sentences.
   - **Why this visualisation** — what about the chart/heatmap/diagram lets a
     human check the claim. Justify the axes, the comparison, what is
     emphasised.
7. **Iterate** until the viz tells the story without the README. The README is
   the safety net, not the argument.

### Boot-check `app.py`

Before declaring done, verify the Gradio app actually starts. A green compute
step with a broken `app.py` is not a complete attempt.

```bash
uv run python app.py &
APP_PID=$!
# Watch stdout (or use the Monitor tool) for "Running on local URL".
kill "$APP_PID"
```

If you only want to check the module constructs without binding a port, run
`uv run python -c "import importlib.util, sys; m=importlib.util.spec_from_file_location('a','app.py'); ml=importlib.util.module_from_spec(m); m.loader.exec_module(ml); print('ok')"`.

## How it is graded

> The rubric is intentionally sparse for v1. More criteria will be added later —
> do not pre-optimise for them; do the v1 things well.

**v1 rubric**

1. **Visual judgement.** The human grader runs
   `uv run python experiments/<goal>/<attempt>/app.py`, interacts with the
   Gradio interface, and decides whether the visualisation makes your claim
   legible. A bar chart that compresses the result into one comparison is often
   better than a 12-panel heatmap.
2. **Visualisation rationale.** The grader reads the *Why this visualisation*
   section of your `README.md`. A strong entry connects the chart choice to
   what the goal asks: the right thing on the y-axis, the right baseline, the
   right grain.

**Coming later (placeholders — do not optimise yet)**

- Quantitative reproducibility of the result across seeds
- Code review for soundness of the interp method
- Cross-attempt comparison within the same goal

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
