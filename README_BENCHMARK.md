# Constructing a goal

You are designing a new mechanistic-interpretability goal under
`experiments/<goal>/`. The framework owns the wiring — recording, storage,
dashboard. **You own the question, the data, and the metric.**

The general principle: *whatever can be computed from an attempt's output
should be a benchmark metric, not a human-judged item.* Move things out of
the rubric in `README_EXPERIMENT.md` whenever you can.

## Three files you write

The goal owns a README, a task, and a benchmark. Every attempt at this goal
imports the second and third instead of duplicating their content — so two
attempts can never disagree on what the data is or how it's scored.

### `experiments/<goal>/README.md`

Documents:

- the **question** the goal asks;
- the **setup** — synthetic generator, real model + dataset, or a mix;
- the **canonical measurement condition** every attempt must use;
- the **payload contract** — exact keys, types, semantics, units (this is
  the shape `task.evaluate` returns and `benchmark.score` consumes);
- the **metrics** — names, formulas, how to read them, bigger-vs-smaller-
  is-better;
- the **bump procedure** when `VERSION` changes.

If a future attempt's author has to read `task.py` or `benchmark.py` to know
what to pass, the README is not detailed enough.

### `experiments/<goal>/task.py`

```python
@dataclass(frozen=True)
class Batch:
    ...                                    # whatever your task needs

def generate(seed: int = 0) -> Batch:
    """Deterministic for a given seed."""
    ...

def evaluate(model_fn) -> dict:
    """Run `model_fn` over a batch, return a payload that benchmark.score consumes."""
    ...
```

Contracts:

- `generate` is **deterministic**: same seed → same batch. If the soft-AND
  setup is fully fixed, `seed` is accepted but ignored; document that.
- `evaluate` takes one argument — the attempt's **model function** — and
  returns the payload dict exactly as `benchmark.score` expects it. Attempts
  never construct the payload themselves; they hand `evaluate` a model and
  receive a ready-to-record payload.
- Pure Python; no I/O, no network.

The `model_fn` signature is the goal's contract with attempts. Document it in
the goal README and keep it narrow — a small typed callable, not a big API
surface.

### `experiments/<goal>/benchmark.py`

```python
VERSION: int                                # bump on incompatible changes

def score(payload: dict) -> dict[str, float | int]:
    ...
```

Constraints:

- Pure Python. No imports from any `<attempt_name>/` directory.
- Deterministic. Same payload → same metrics.
- Side-effect free. No file I/O, no network, no time-dependent values.
- Defensive on its inputs. Raise `ValueError` / `KeyError` with a clear
  message when the contract is violated — silent garbage is worse than a
  hard failure.

## Designing the payload

The payload is the *minimum* data needed to compute every metric. Three
principles:

1. **Model-agnostic.** Any attempt at this goal — current or hypothetical —
   should be able to produce the payload. Don't bake in the structure of one
   approach. If you can imagine two attempts that would naturally hand over
   different shapes, you've under-specified.
2. **Pre-aggregated.** Don't pass raw tensors. Pass scalars or small dicts.
   The attempt's `main.py` does the reduction; `score()` just combines.
3. **Self-describing.** Include the configuration used (canonical scale,
   dataset slice, model name) even if `score()` doesn't read it. Future-you
   debugging a regression at 3 a.m. will thank you.

For multi-condition benchmarks (a sweep across a parameter), use a list of
records:

```python
sweep: list[{
    "<axis>": <value>,
    "<measurement_a>": ...,
    "<measurement_b>": ...,
}]
```

This generalises beyond one knob — you can add a second axis later without a
contract change as long as the per-record shape grows monotonically.

## Designing the metrics

Return a flat dict of named scalars. Aim for three classes per benchmark:

- **One headline summary.** A single number an attempt should optimise. Goes
  on the leaderboard. For `attention_and`, this is `superposition_robustness`.
  If you can't write down what the one number is, the goal isn't sharp enough.
- **Per-slice values.** One scalar per condition in your sweep, so the panel's
  metric dropdown lets the grader investigate where the method holds and
  where it breaks. Naming pattern: `<metric>_<axis>_<value>` with floats in
  `0p7`-form (`and_sharpness_cos_0p7`).
- **Reference baselines.** Measure the strawman under the same conditions
  (`linear_baseline_sharpness_*`). A method *beating* the baseline is
  meaningful; the same method in isolation is not.

Always include `version` as the first key. The dashboard filters to the
highest version present, so old runs stay legible without polluting the
active series.

## Naming conventions

| pattern | use |
|---------|-----|
| `<metric>_canonical`            | the headline value at the default condition |
| `<metric>_<axis>_<value>`       | per-slice value; floats use `0p7` not `0.7` |
| `<thing>_robustness`            | a ratio across a sweep, ideally in `[0, 1]` |
| `linear_baseline_<metric>`      | the no-mechanism reference, same conditions |
| `lift_over_<baseline>`          | your method minus the baseline, same units |

Use consistent direction-of-better across a goal — if `and_sharpness` is
bigger-is-better, don't make `failure_rate` smaller-is-better in the same
file. The dashboard doesn't render directionality; the grader has to infer
it from the name.

## Bumping `VERSION`

You **must** bump when:

- you change the formula of any existing metric;
- you rename, remove, or retype a payload key;
- you change the canonical condition (e.g. canonical scale).

You **don't** need to bump when:

- you add a new metric without changing existing ones;
- you add an optional payload key with a default;
- you add a slice to a sweep that was already extensible.

After bumping: update the goal's `README.md` benchmark contract in the same
commit. Old `benchmark.json` files stay on disk — the dashboard hides them by
default and shows a count of older runs in the header.

## What good metrics look like

- **Scalar.** No nested structures, no lists. Each metric is one float or int.
- **Comparable across attempts.** Different attempts produce different
  numbers; the metric ranks them sensibly. A metric that ties every attempt
  isn't doing work.
- **Has a sensible range.** Unbounded metrics (`inf` when the denominator is
  0) need explicit handling — either pick a different formulation or document
  the edge case.
- **Stable to nuisance variation.** Two near-identical runs of the same
  attempt should produce nearly the same metric. If they don't, you're
  measuring noise.

## Anti-patterns

- **One bloated metric.** Bundling several signals into a single weighted sum
  ("score = 0.4·sharpness + 0.3·robustness + …") obscures what's being
  measured and forces the weights to be politically defended. Return them
  separately and let the leaderboard show all of them.
- **Per-attempt branches.** `score()` must not branch on the attempt name or
  read attempt-specific files. The payload is the only interface.
- **Tensors in the payload.** Never. Reduce in `main.py`.
- **Silent contract changes.** Always bump `VERSION` when keys move. The
  panel and the human grader both rely on it.
- **Metric inflation.** Twenty metrics is rarely better than five. Each
  metric should answer a distinct question; if two move together every time,
  one of them is redundant.

## Optional pipeline hooks

`benchmark.py` may also export two pipeline-only knobs. Both are optional —
omit them and the pipeline uses sensible defaults.

### `GPU_REQUIREMENT: int = 1`

How many GPU slots the experiment subprocess needs. **The minimum is 1** —
every attempt runs on the GPU and the pipeline clamps anything lower (or
missing) up to 1, so `0` is meaningless. Set it to `2` only for goals doing
big-model activation patching or model-parallel training. The pipeline
acquires that many slots from the file-locked pool before launching `main.py`
and boot-checking `app.py`, and passes `CUDA_VISIBLE_DEVICES` accordingly. The
pool size itself is `AGENTIC_GPU_COUNT` (default 2).

Attempts are launched through a GPU guard that runs `main.py` and then asserts
it actually allocated CUDA memory; a pure-CPU/NumPy attempt is rejected and
retried. This only constrains *attempts* — your `task.py` and `benchmark.py`
stay pure CPU/NumPy (the smoke test runs them without a GPU).

### `is_obviously_broken(metrics: dict) -> bool`

The pipeline calls this **right after** the solver's `main.py` finishes,
passing the `metrics` dict from the just-written `benchmark.json`. If it
returns `True`, the pipeline marks the attempt `failed` and **skips the
jury** — the jury is the most expensive stage and running it on a clearly
degenerate attempt is pure waste.

Use it for things the goal author can detect mechanically without a model in
the loop: NaN/inf math failures, metrics worse than the no-mechanism
baseline, normalisation that didn't sum to 1, etc. The predicate must never
return `True` for a borderline-but-real result — it only ever short-circuits
the jury, never replaces it for the affirmative side.

Example, from `experiments/attention_and/benchmark.py`:

```python
def is_obviously_broken(metrics: dict) -> bool:
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    sharp = metrics.get("and_sharpness_canonical")
    baseline = metrics.get("linear_baseline_sharpness_cos_0p0")
    if isinstance(sharp, int | float) and isinstance(baseline, int | float):
        if sharp <= baseline * 1.5:
            return True
    return False
```

## Worked example

The simplest concrete reference is `experiments/attention_and/benchmark.py`
and its goal `README.md`:

- The payload is a `sweep` over `cos(q_A, q_B)` plus a few labels.
- Per-slice metrics name the cosine in the key (`and_sharpness_cos_0p7`).
- The headline summary is `superposition_robustness`.
- A linear-baseline metric is computed under identical conditions for
  contrast.
- `VERSION = 2` because the original v1 only measured at the orthogonal
  anchor; old v1 `benchmark.json` files stay on disk but the panel filters
  them out.

Read it before writing your first one.
