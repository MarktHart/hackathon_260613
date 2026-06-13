# Attention Constraint Propagation (Bracket Matching)

## Question

When a transformer processes sequences containing nested syntactic
constraints — matched open/close bracket pairs — do its attention heads
*propagate* those constraints by attending from one bracket to its partner?
And how does that fidelity vary with the **positional distance** between the
two constrained tokens?

This goal tests whether an interpretability method (or a model) produces
attention that **aligns with the ground-truth constraint graph**: for each
matched pair `(open_pos, close_pos)`, the relevant head should place weight on
the partner position. We measure this alignment per head and sweep it across a
range of open↔close distances.

## Setup

**Synthetic generator (`task.generate`).** We build a batch of sequences over
a small vocabulary:

- Filler tokens: ids `0..99` (`N_FILLER = 100`).
- Two bracket *types* (`CONSTRAINT_TYPES = 2`):
  - Type A: `OPEN_A = 100`, `CLOSE_A = 101`
  - Type B: `OPEN_B = 102`, `CLOSE_B = 103`
- `VOCAB_SIZE = 104`.

Each sequence is filled with random filler tokens, then 2–4 matched pairs of
each bracket type are placed at distances drawn from
`DISTANCES = (1, 2, 4, 8, 12, 16)`, never overlapping. For every placed pair
`(o, c)` with `d = |o - c|`, the generator records **two directed entries**,
`(o, c, d)` and `(c, o, d)`, in the per-sequence ground-truth constraint list.
A pair that cannot be placed without overlap after 40 attempts is silently
skipped (so the realised entry count per distance varies slightly, but is
always large at the default scale).

Because the canonical distance (4) is one of the fixed `DISTANCES`, it is
guaranteed to appear in every sweep regardless of seed.

**No model is shipped with the goal.** The goal defines only the data and the
scoring. Each attempt supplies its own `model_fn` (a real transformer, a probe,
a hand-built circuit — anything that maps tokens to attention weights).
`task.random_model_fn()` provides a pure-NumPy uniform-attention baseline used
by the smoke test and as the no-mechanism reference.

**`model_fn` contract** (the contract between attempts and this goal):

```python
def model_fn(input_ids: np.ndarray) -> np.ndarray:
    """
    Args:
        input_ids: int array of shape (batch, seq_len), token ids in [0, VOCAB_SIZE).
    Returns:
        attn: float32 array of shape (batch, n_layers, n_heads, seq_len, seq_len).
              Post-softmax attention weights. attn[b, l, h, i, j] is the weight
              that query position i places on key position j, for layer l head h.
    """
```

`n_layers` and `n_heads` are **model-dependent** and inferred from the returned
array's shape — an attempt may return any number of layers/heads. All arrays
are NumPy. The attempt's `main.py` calls `task.evaluate(model_fn)` and records
the returned payload; attempts never build the payload themselves.

## Canonical Measurement Condition

Fixed in `task.py` and used by `evaluate` when no batch is passed:

- Sequence length: `SEQ_LEN = 32`
- Number of sequences: `NUM_SEQUENCES = 500`
- Constraint types: `CONSTRAINT_TYPES = 2`
- Distance sweep: `DISTANCES = (1, 2, 4, 8, 12, 16)`
- **Canonical distance: `CANONICAL_DISTANCE = 4`** (the headline slice)
- Vocabulary size: `VOCAB_SIZE = 104`
- Seed: `SEED = 0`

## Payload Contract

`task.evaluate(model_fn)` returns a `dict` with exactly these keys:

```python
{
    "version": 1,                       # int, must match benchmark.VERSION
    "config": {
        "seq_len": 32,
        "num_sequences": 500,
        "constraint_types": 2,
        "canonical_distance": 4,
        "seed": 0,
    },
    "model_info": {                     # inferred from model_fn output shape
        "n_layers": <int>,
        "n_heads": <int>,
    },
    "sweep": [                          # one record per distance present in the data
        {
            "distance": 4,             # int, the open<->close distance for this slice
            "n_entries": <int>,        # number of directed constraint entries at this distance
            "heads": [                 # one entry per (layer, head)
                {"layer": 0, "head": 0, "alignment": <float>},
                ...
            ],
            "mean_alignment": <float>, # mean over all (layer, head) alignments
            "max_alignment": <float>,  # max over all (layer, head) alignments
            "best_head": {             # the (layer, head) achieving max_alignment
                "layer": <int>, "head": <int>, "alignment": <float>,
            },
        },
        ...
    ],
}
```

**Semantics:**

- **Alignment** for a `(layer, head)` at distance `d` is the mean, over all
  directed constraint entries `(i, j, d)` in the batch, of the attention weight
  `attn[b, l, h, i, j]` — i.e. how much weight that head places on the *partner*
  position of each constrained token. Higher = the head tracks the constraint
  more strongly. A uniform-attention head scores exactly `1/seq_len` at every
  distance (the random baseline).
- `mean_alignment` / `max_alignment` aggregate across heads within a distance
  slice; `best_head` names the single most-aligned head.
- The `sweep` contains one record for every distinct distance that actually
  appears in the generated constraint graph (always a subset of `DISTANCES`).

## Metrics

`benchmark.score(payload)` returns a flat `dict[str, float | int]`. Let
`baseline = 1 / seq_len` (the uniform-attention alignment).

| Metric | Formula | Direction | Interpretation |
|--------|---------|-----------|----------------|
| `version` | `VERSION` | — | Contract version |
| `canonical_distance` | `config["canonical_distance"]` | — | Which slice is the headline |
| `baseline_alignment_canonical` | `1 / seq_len` | — | Uniform-attention reference |
| `mean_alignment_dist_<d>` | `sweep[d].mean_alignment` | ↑ | Mean head alignment at distance d |
| `max_alignment_dist_<d>` | `sweep[d].max_alignment` | ↑ | Best head alignment at distance d |
| `random_baseline_alignment_dist_<d>` | `1 / seq_len` | — | Per-slice random reference |
| `max_head_alignment_canonical` | `sweep[canonical].max_alignment` | ↑ | Best head alignment at canonical distance |
| `best_head_layer_canonical` | `sweep[canonical].best_head.layer` | — | Layer of the best canonical head |
| `best_head_head_canonical` | `sweep[canonical].best_head.head` | — | Head index of the best canonical head |
| `constraint_propagation_fidelity` | `max_head_alignment_canonical / baseline` | ↑ | **Headline.** Best-head alignment as a multiple of random |

**Headline summary metric:** `constraint_propagation_fidelity` — the best
head's alignment at the canonical distance, expressed as a multiple of the
uniform-attention baseline. `1.0` means "no better than random"; `> 1.0` means
the model concentrates attention on partner positions; `< 1.0` means it
actively avoids them.

**Per-slice values:** `mean_alignment_dist_<d>`, `max_alignment_dist_<d>`, and
`random_baseline_alignment_dist_<d>` for every distance `d` in the sweep, so the
panel can show *where* constraint tracking holds up and where it decays with
distance.

**Baseline:** `baseline_alignment_canonical` / `random_baseline_alignment_dist_<d>`
are the uniform-attention alignment `1/seq_len`, computed under identical
conditions. A method is meaningful only when its alignment exceeds this.

**Edge cases handled by `score`:**

- `seq_len <= 0` → `ValueError` (also guards the `1/seq_len` denominator).
- Missing required keys (`version`, `config`, `sweep`) → `KeyError`.
- Wrong version → `ValueError`.
- `baseline == 0` → fidelity set to `0.0` instead of dividing.
- Canonical distance absent from the sweep (empty / partial sweep) → fidelity
  `0.0`, best head reported as `(-1, -1)`; no crash.
- `is_obviously_broken(metrics)` returns `True` on any NaN/inf metric, or when
  `constraint_propagation_fidelity <= 1.0` (at-or-below random carries no
  mechanism worth jurying).

## Bump Procedure

Bump `VERSION` (in **both** `benchmark.py` and this README) when:

- Payload keys are added, removed, or retyped.
- Any existing metric formula changes.
- The canonical condition changes (`SEQ_LEN`, `CANONICAL_DISTANCE`,
  `DISTANCES`, `CONSTRAINT_TYPES`, vocabulary, or seed).

Do **not** bump when adding a new optional payload key with a default, or a new
metric that does not alter existing ones, or a new distance slice (the sweep is
already extensible). After bumping, update this README's payload/metric tables
in the same commit; old `benchmark.json` files stay on disk and the dashboard
filters to the highest version present.
