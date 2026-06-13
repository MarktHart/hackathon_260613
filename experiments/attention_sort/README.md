# attention_sort

## The question

Does an attention head implement **sorting** — routing each output slot to the
input token it should read in sorted order — and does that mechanism hold up as
the sequence gets longer?

Concretely: given a sequence of `L` distinct values, a sorting head's output
slot `i` should attend to the input position holding the **i-th smallest**
value. We measure how well an attempt's attention realises this permutation,
and how the quality decays as `L` grows. A head that has genuinely learned to
sort generalises across lengths; a positional shortcut (e.g. "always attend
left-to-right") collapses on longer sequences.

## Setup

**Fully synthetic.** No trained model, no dataset. `task.generate(seed)` draws,
for each sequence length `L` in the sweep, `N_SEQUENCES` sequences of `L`
continuous values in `[0, 1)` (distinct with probability 1). The ground-truth
sorting permutation for a sequence is `argsort(values)`: output slot `i` should
attend to input position `argsort(values)[i]`.

`generate` is deterministic — same `seed` → same data. The canonical evaluation
uses a fixed internal seed (`EVAL_SEED = 42`), so every attempt is scored on
identical sequences.

## Canonical measurement condition

| knob              | value                  |
|-------------------|------------------------|
| sweep lengths     | `(4, 8, 16, 32)`       |
| canonical length  | `8`                    |
| sequences / length| `256`                  |
| value distribution| `uniform[0, 1)`        |
| eval seed         | `42`                   |

All per-length metrics are reported across the sweep; the headline and
`*_canonical` metrics use length `8`.

## model_fn contract

Attempts hand `task.evaluate` a single callable:

```python
ModelFn = Callable[[np.ndarray], np.ndarray]
#   values: np.ndarray[B, L] float32   -> a batch of sequences for ONE length
#   returns: np.ndarray[B, L, L] float -> attention logits
```

Row `i` of the returned `[L, L]` matrix is the (unnormalised) logits over input
positions for output slot `i`. `evaluate` applies a row-wise softmax, so an
attempt may return either raw logits or an already row-stochastic matrix — the
softmax is monotone and preserves the argmax / mass structure that is scored.

`evaluate` calls `model_fn` once per length in the sweep. Pure NumPy in, pure
NumPy out — no torch tensors crossing the boundary, no GPU dependency inside the
callable's data contract. (The attempt's `main.py` may still use a GPU to
*produce* the callable; `task.py` / `benchmark.py` stay pure CPU/NumPy.)

`task.random_model_fn() -> ModelFn` returns a reference callable emitting random
logits — used by the pipeline smoke test.

## Payload contract

`task.evaluate(model_fn)` returns:

```python
{
  "version": 1,                       # int, must equal benchmark.VERSION
  "task": "attention_sort",           # str label
  "canonical_length": 8,              # int
  "sweep_lengths": [4, 8, 16, 32],    # list[int], ascending
  "n_sequences": 256,                 # int, sequences per length
  "sweep": [                          # one record per length
    {
      "length": int,                  # sequence length L
      "sort_accuracy": float,         # frac of output slots whose argmax key == target key
      "target_mass": float,           # mean attention mass on the correct target key
      "output_sortedness": float,     # frac of adjacent output pairs in non-decreasing order
      "unsorted_sortedness": float,   # same, for the raw (unsorted) input — "did nothing" baseline
      "uniform_accuracy": float,      # 1/L, expected argmax-hit of uniform/random attention
      "n_sequences": int,
    },
    ...
  ],
}
```

`benchmark.score` consumes only `version`, `canonical_length`, and `sweep`; the
rest is self-describing context for debugging.

## Metrics

All metrics are **bigger-is-better**. `version` is the first key (the dashboard
filters to the highest version present).

| metric                                 | meaning |
|----------------------------------------|---------|
| `sort_robustness`                      | **headline.** `sort_accuracy` at the longest length ÷ at the shortest, clamped to `[0, 1]`. Captures how well the sorting mechanism generalises across length. |
| `sort_accuracy_canonical`              | argmax-key accuracy at length 8. |
| `output_sortedness_canonical`          | sortedness of the attention-mixed output at length 8. |
| `target_mass_canonical`                | mean mass placed on the correct key at length 8. |
| `sort_accuracy_mean`                   | mean `sort_accuracy` across the whole sweep. |
| `sort_accuracy_len_<L>`                | per-slice argmax-key accuracy at length `L`. |
| `output_sortedness_len_<L>`            | per-slice output sortedness at length `L`. |
| `target_mass_len_<L>`                  | per-slice target mass at length `L`. |
| `unsorted_baseline_sortedness_len_<L>` | sortedness of the raw unsorted input at length `L` (~0.5 reference). |
| `uniform_baseline_accuracy_canonical`  | `1/8` — uniform/random attention argmax accuracy at the canonical length (reference). |
| `lift_over_unsorted_canonical`         | `output_sortedness_canonical − unsorted_sortedness` at length 8. Positive ⇒ the head actually sorts. |

### Reading them

- A genuine sorting head: `sort_accuracy_canonical` near 1, `sort_robustness`
  near 1, `lift_over_unsorted_canonical` clearly positive (toward `1 − 0.5 ≈
  0.5`).
- A positional shortcut: high `sort_accuracy_len_4`, decaying
  `sort_accuracy_len_32` ⇒ low `sort_robustness`.
- Random / uniform attention: `sort_accuracy ≈ 1/L`, `lift_over_unsorted ≈ 0`.

### Pipeline hooks

- `GPU_REQUIREMENT = 1` — attempts run on the GPU.
- `is_obviously_broken(metrics)` — `True` on any NaN/inf, or when
  `sort_accuracy_canonical` fails to clearly beat the uniform baseline
  (`≤ max(2 · uniform_accuracy, 0.05)`). Only ever short-circuits the jury for
  mechanically degenerate attempts.

## Bump procedure

Bump `benchmark.VERSION` (and `payload["version"]` in `task.evaluate`) when you:

- change any existing metric's formula;
- rename / remove / retype a payload key;
- change the canonical condition (sweep lengths, canonical length, eval seed,
  value distribution).

You need **not** bump when adding a new metric, an optional payload key with a
default, or an extra length to the (already extensible) sweep. Update this
README's contract in the same commit as any bump. Old `benchmark.json` files
stay on disk; the dashboard filters to the highest version.
