# Suggested changes to `README_EXPERIMENT.md`

Context: the current scaffolding is shaped around "interpret a known model."
[`BLOCKS.md`](BLOCKS.md) shifts the work so that, for most entries, **the
model has to be built first** (trained, or for easy blocks, hand-constructed),
the algorithms compose into a DAG, and many attempts will share the same
underlying weights. Three things the current scaffolding does not handle:

1. **Model construction as a first-class step** (training *or* hand-built
   weights), not an assumption.
2. **The data generator** that defines a block — currently only prose in a
   goal README.
3. **Reuse and comparison** across attempts and across blocks.

What follows is the set of changes I would make, roughly in priority order.

---

## 1. Promote "task" to a first-class artefact

Add `experiments/<block>/task.py` exposing a single data generator, a
reference solver, and a scoring function. Every attempt imports it.

Without this, two attempts on the same block train on subtly different
distributions and aren't comparable. It is also a prerequisite for the
hand-construction route (#7 below), which needs the generator to
self-verify.

## 2. Add a quantitative gate before the visual claim

The current rubric only grades the Gradio chart. For BLOCKS, "did the model
actually learn the algorithm" is a prerequisite — surface it as one number
(test-set accuracy or per-token loss vs. the reference solver) that the
grader sees alongside the chart. Move "quantitative reproducibility" from
"coming later" to v1.

## 3. Separate trained-model registry from per-attempt results

Put model weights under `experiments/<block>/models/<config-hash>/` and let
attempts load them. Tag each entry with `provenance` (`trained` /
`hand_built`). Otherwise every interp attempt re-trains, weights drift
between attempts, and circuits cannot be compared at all. `results/<run-id>/`
should hold *interp artefacts*, not weights.

## 4. Number the blocks; use `BLOCKS.md` as the curriculum index

Name directories `experiments/02_xor/`, `experiments/18_induction/` so `ls`
shows the tier. Each `<block>/README.md` cites its `BLOCKS.md` entry and its
`builds on:` dependencies. The block spec should state: data shape, success
metric, suspected minimal architecture, hand-construction feasibility
(see #7), and which earlier blocks' models can be transferred or probed
against.

## 5. Make cross-attempt comparison the default view, not deferred

Add `experiments/<block>/SUMMARY.md` (auto-generated) listing attempts ×
(test accuracy, smallest config that solved it, provenance, link to viz).
With dozens of attempts per block this is the only way the user can
navigate them. Add `experiments/SUMMARY.md` at the top level showing
curriculum progress across blocks.

## 6. Name the attempt archetypes

Right now "attempt = hypothesis" is open-ended; for BLOCKS most attempts
fall into a small set:

- `hand_built` — construct weights directly (see #7)
- `train_baseline` — smallest config that learns it
- `min_config_search` — sweep size/depth/heads for the floor
- `head_ablation` / `path_patching` — circuit attribution
- `probe_<feature>` — read out a hypothesised feature
- `transfer_from_<earlier_block>` — load weights from a builds-on
  dependency, fine-tune or probe

Listing these in the README cuts decision overhead and makes the structure
predictable.

## 7. Hand-constructed weights as a first-class attempt type

For Tier 0–3 blocks, the agent can usually write the weights directly
(textbook constructions exist for AND/OR/XOR, shift-by-k, previous-token,
argmax, counting, span attention, bracket-depth, induction, reversal,
small-k sort). This is more than another archetype — it changes the rubric.

- The hand-built model is a **ground-truth circuit**. The interp question
  becomes "does the trained model implement the same circuit?" — concrete
  and falsifiable, instead of "what is this trained model doing?".
- The strongest visualisation on easy blocks is a **side-by-side** of
  trained vs. hand-built weights (attention pattern, OV/QK matrices). If
  they match, the interp claim is essentially proven. If they don't, that
  is the interesting finding: SGD found a different circuit, and now you
  have a target for "why."
- Per-block annotation: `hand-construction feasible? yes / sketched / no`.
  Yes through roughly Tier 3; mixed at Tier 4 (FSM, addition-with-carry
  are tedious but doable); generally no from Tier 5 onward (DP tables
  become a paper, not an attempt).
- Caveat to state in the README: a hand-built solution proves the task is
  *expressible* at a given size, not that SGD will find it. The gap
  between expressible and findable is itself a worthwhile sub-question on
  blocks where training fails.

## 8. Loosen "one attempt = one hypothesis" at the train/interp seam

A clean workflow is: one attempt trains (or hand-builds) and registers
the model, with the claim "this config learns the task at >X% acc"; later
attempts interpret it. Either make this two-phase split explicit, or keep
attempts bundled but require any interp attempt to declare which
registered model it analyses.

## 9. Update "what to visualise" guidance

For BLOCKS the natural axis is *ablation severity vs. task accuracy*,
which doubles as the chart and as the quantitative gate from #2. On
hand-construction-feasible blocks, add: *trained-vs-hand-built weight
overlay* as the default first chart.

---

## What to leave alone

- Two-level layout (`<block>/<attempt>/`).
- Shared `pyproject.toml` as single source of truth.
- Gradio `app.py` for visual judgement, defaulting to the most recent run.
- The per-attempt README "what I did / why this viz" structure.
- "Smallest artefact that, if flipped, would change the claim" as the
  guiding principle for visualisation — it still applies, only the
  artefact shifts (often to an ablation curve or a weight overlay).
