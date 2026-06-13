# Attention Deduplication Goal

## Question

Does an attention mechanism **route a duplicate token's query back to the
position of its previous occurrence?** This is the behaviour of a
*duplicate-token head* — a canonical mechanistic-interpretability motif and a
building block of induction. We measure, for every position holding a repeated
token, how much attention mass lands on the **most recent earlier position with
the same token id**, and how often that position is the arg-max key.

This is a fully synthetic benchmark with known ground-truth duplication
structure — no trained model required.

---

## Setup

**Synthetic generator only.** Each slice is a batch of integer token sequences.
Tokens are sampled so that a controlled fraction of positions are *duplicates*
of an earlier token; the rest are *first-seen*. For every position the generator
records `prev` — the index of the same token's previous occurrence, or `-1` if
first-seen.

- **Sequence length**: `L = 24`
- **Sequences per slice**: `N = 64`
- **Vocab size**: `V = 64`
- **Sweep axis** — duplicate density `dup_rate ∈ {0.1, 0.3, 0.5, 0.7}`
- **Canonical condition**: `dup_rate = 0.5`, `seed = 0`

`generate(seed)` is deterministic: same seed → identical `Batch`. The canonical
condition uses `seed = 0`; non-zero seeds reshuffle the token streams but keep
the same shapes and sweep axis.

---

## Canonical Measurement Condition

Every attempt **must** be evaluated through `task.evaluate(model_fn)`, which runs
on `generate(0)` across the full dup-rate sweep. No other data, no other
hyperparameters. Attempts only supply a `model_fn`.

---

## Model Function Signature

```python
def model_fn(tokens: np.ndarray) -> np.ndarray:
    """
    Args:
        tokens: np.ndarray[int, (N, L)]   token ids for one slice.

    Returns:
        attn: np.ndarray[float, (N, L, L)]  causal attention weights.
              attn[s, q, k] = weight from query q to key k (k <= q).
              Rows should be row-stochastic over causal keys.
    """
```

- `N = 64`, `L = 24` (fixed by the generator).
- `model_fn` is called **once per slice** (4 times total).
- Non-causal entries (`k > q`) are zeroed and each row is renormalised by the
  evaluator, so the metric rewards exactly one thing: putting mass on the
  previous occurrence. Empty rows fall back to uniform-causal.
- The attempt may compute attention however it likes (trained head, analytic,
  random). The benchmark only scores the returned array.

---

## Payload Contract

`task.evaluate(model_fn)` returns a `dict` with exactly these keys:

```python
{
    "version": 1,                  # matches benchmark.VERSION
    "task": "attention_dedupe",
    "seed": 0,
    "seq_len": 24,
    "vocab_size": 64,
    "n_seqs": 64,
    "dup_rates": [0.1, 0.3, 0.5, 0.7],
    "canonical_dup_rate": 0.5,
    "sweep": [                     # one record per dup_rate
        {
            "dup_rate": 0.1,
            "n_dup_positions": int,        # # duplicate query positions in slice
            "n_first_seen": int,           # # first-seen positions in slice
            "dedup_mass": float,           # mean attn mass on previous occurrence
            "dedup_accuracy": float,       # fraction where arg-max key == prev occ
            "first_seen_self_mass": float, # mean self-attention at first-seen toks
            "baseline_dedup_mass": float,  # uniform-causal mass on the target
            "baseline_dedup_accuracy": float,  # = baseline_dedup_mass (uniform)
        },
        ...
    ],
}
```

All `sweep` measurements are in `[0, 1]`. The attempt **never constructs this
dict** — `task.evaluate` builds it from the raw attention arrays.

---

## Metrics

`benchmark.score(payload)` returns a flat `dict[str, float | int]`. Every metric
is **bigger-is-better**.

| Metric | Meaning |
|--------|---------|
| `version` | `benchmark.VERSION` (always first key) |
| `dedup_robustness` | **Headline.** Mean `dedup_mass` across the sweep, clamped to `[0, 1]`. The single number to optimise. |
| `dedup_mass_canonical` | `dedup_mass` at `dup_rate = 0.5`. |
| `dedup_accuracy_canonical` | `dedup_accuracy` at `dup_rate = 0.5`. |
| `dedup_accuracy_mean` | Mean `dedup_accuracy` across the sweep. |
| `first_seen_self_mass_canonical` | Mean self-attention on first-seen tokens at canonical (a good dedupe head leaves novel tokens on the diagonal). |
| `uniform_baseline_dedup_mass_canonical` | Uniform-causal reference mass at canonical. |
| `lift_over_uniform_canonical` | `dedup_mass_canonical − uniform_baseline_dedup_mass_canonical`. |
| `dedup_mass_rate_<v>` | Per-slice `dedup_mass`, e.g. `dedup_mass_rate_0p7`. |
| `dedup_accuracy_rate_<v>` | Per-slice `dedup_accuracy`. |
| `baseline_dedup_mass_rate_<v>` | Per-slice uniform-causal reference. |

Slice keys use `0p7`-form floats. A method beating the uniform baseline is
meaningful; the raw number in isolation is not.

---

## Bump Procedure

Increment `VERSION` in `benchmark.py` when:
- any metric formula changes,
- payload keys are added/removed/retyped,
- the canonical condition (dup_rate, seq_len, vocab, seed) changes.

Adding a new metric without touching existing ones, or adding a slice to the
already-extensible sweep, does **not** require a bump. After a bump, update this
README's Payload Contract and Metrics tables in the same commit. Old
`benchmark.json` files stay on disk; the dashboard filters to the highest
`version`.

---

## Optional Pipeline Hooks

- `GPU_REQUIREMENT = 1` (default) — attempts run on GPU; `task.py`/`benchmark.py`
  stay pure CPU/NumPy.
- `is_obviously_broken(metrics)` — returns `True` if any metric is NaN/inf, or if
  `dedup_mass_canonical <= uniform_baseline_dedup_mass_canonical` (no
  deduplication mechanism present), skipping the jury.
