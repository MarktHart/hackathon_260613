# attention_equality

## The question

Does an attention head implement an **equality lookup** — routing its
attention onto earlier key positions that hold the *same* token as the query?

This is the atom under induction heads and copy/lookup circuits: before a head
can copy "what followed the previous occurrence of X", it must first *find* the
previous occurrence of X — i.e. attend to the key whose token **equals** the
query's token. We measure that routing in isolation.

## Setup (synthetic)

Pure-NumPy synthetic generator; no trained model, no GPU.

Each sequence plants exactly **one** equal pair. A target token `t` is placed
at two positions `p1 < p2`; every other position holds a **distinct**
distractor token (`!= t`, and distinct from one another), drawn without
replacement from a vocabulary of `V = 128`. Because all other symbols are
unique, `(p1, p2)` is the *only* equal pair in the sequence.

We treat `p2` as the query of interest and ask how much attention mass the head
places on the single matching key `p1`:

```
match_mass(sequence) = attn[p2, p1]      # in [0, 1]
```

- A perfect equality head → `match_mass ≈ 1.0`.
- A uniform-attention head → `match_mass ≈ 1/(p2 + 1)` (it spreads mass over
  all `p2 + 1` causally-allowed keys).

The **difficulty axis** is sequence length `L`: more positions means more
distractors and a harder lookup. The uniform baseline shrinks with `L`, so
beating it at larger `L` is the meaningful signal.

### Canonical measurement condition

| parameter        | value                  |
|------------------|------------------------|
| vocabulary `V`   | 128                    |
| sequences `B`    | 256                    |
| canonical `L`    | 16                     |
| `L` sweep        | `[8, 16, 32, 64]`      |
| masking          | causal (incl. self)    |
| seed             | 0 (fixed in evaluate)  |

`generate(seed, L)` is deterministic for a given `(seed, L)`. `evaluate` always
uses `seed=0` and sweeps the `L` axis above.

## The `model_fn` contract

An attempt hands `task.evaluate` a single callable:

```python
def model_fn(batch: Batch) -> np.ndarray:
    """
    batch.tokens : (B, L) int32   token ids
    batch.mask   : (B, L, L) bool causal mask (lower-triangular incl. diagonal)
    batch.p1     : (B,) int        earlier position of the planted equal pair
    batch.p2     : (B,) int        later position (the query of interest)
    batch.L, batch.V : ints

    returns attn : (B, L, L) float, row-stochastic over the causally-allowed
                   keys (mass 0 on disallowed positions; each query row sums ~1).
    """
```

Attempts never build the payload themselves — they implement `model_fn` and
receive a ready-to-record payload from `evaluate`.

## Payload contract

`task.evaluate(model_fn)` returns:

```python
{
  "version": 1,
  "config": {                       # self-describing; score() does not read it
      "V": 128, "B": 256,
      "canonical_L": 16,
      "L_sweep": [8, 16, 32, 64],
      "causal": True,
  },
  "canonical": {                    # the L == 16 slice, duplicated for convenience
      "L": 16, "n_eval": 256,
      "match_mass": float,          # mean attn[p2, p1]              (bigger better)
      "uniform_baseline": float,    # mean 1/(p2+1)                  (reference)
      "attn_rowsum_max_dev": float, # max |row sum - 1| over allowed keys
  },
  "sweep": [                        # one record per L in L_sweep
      { "L": int, "n_eval": int,
        "match_mass": float,
        "uniform_baseline": float,
        "attn_rowsum_max_dev": float },
      ...
  ],
  "attn_rowsum_max_dev": float,     # max over all slices (sanity diagnostic)
}
```

All values are pre-aggregated scalars — no tensors cross the boundary.

## Metrics (`benchmark.score`)

`version` is always the first key. The dashboard filters to the highest
version present.

| metric                          | meaning                                              | better |
|---------------------------------|------------------------------------------------------|--------|
| `equality_robustness`           | **headline** — mean `match_mass` across the L sweep  | bigger |
| `match_mass_canonical`          | `match_mass` at `L = 16`                              | bigger |
| `match_mass_L_<L>`              | per-slice `match_mass` (e.g. `match_mass_L_32`)       | bigger |
| `uniform_baseline_L_<L>`        | uniform reference at that slice                       | —      |
| `uniform_baseline_canonical`    | uniform reference at `L = 16`                         | —      |
| `uniform_baseline_robustness`   | mean uniform reference across the sweep               | —      |
| `lift_over_uniform_L_<L>`       | `match_mass - uniform_baseline` at that slice         | bigger |
| `lift_over_uniform_canonical`   | lift at `L = 16`                                      | bigger |
| `attn_rowsum_max_dev`           | max deviation of row sums from 1 (sanity)             | smaller|

**Headline to optimise:** `equality_robustness` ∈ `[0, 1]`. A method beating
`uniform_baseline_robustness` across the sweep is doing real equality routing;
matching it is not.

### Pipeline hooks

- `GPU_REQUIREMENT = 0` — pure NumPy, no GPU slot needed.
- `is_obviously_broken(metrics)` returns `True` (skips the jury) when: any
  metric is NaN/inf; `attn_rowsum_max_dev > 0.1` (attention not row-stochastic
  → malformed); or `match_mass_canonical <= 1.5 × uniform_baseline_canonical`
  (no meaningful lift over uniform). It never returns `True` for a
  borderline-but-real result.

## Bump procedure

`VERSION` (in `benchmark.py`) and the payload `version` are currently `1`.
Bump both — and update this contract in the same commit — when you: change the
formula of any existing metric; rename/remove/retype a payload key; or change
the canonical condition (`V`, `B`, canonical `L`, or the `L` sweep). Adding a
new metric or an extra `L` slice does **not** require a bump (the sweep is
already extensible and per-slice keys are additive). Old `benchmark.json` files
stay on disk; the dashboard hides superseded versions.
