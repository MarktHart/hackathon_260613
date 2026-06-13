# attention_matrix_chain

## Question

Two stacked attention layers compose: the **effective two-hop attention** a
token receives is the matrix product of the per-layer patterns,
`A_chain = A2 @ A1`. This is the "virtual attention head" view at the heart of
attention-circuit analysis (e.g. induction heads).

**Can a mechanism reconstruct the composed attention matrix `A_chain` from the
two single-layer patterns `A1`, `A2`?** And does it keep doing so as the
attention rows become more *peaked* — the regime where composition genuinely
matters and a single-hop shortcut (using `A2` alone) breaks down?

## Setup

**Synthetic generator** — fully controlled, no trained models, pure NumPy.

For each condition we sample two row-stochastic attention patterns `A1`, `A2`
of shape `(num_heads, seq_len, seq_len)`. Each attention row is drawn from a
`Dirichlet(alpha)` distribution; the ground-truth composed pattern is the
per-head matrix product

```
A_chain[h] = A2[h] @ A1[h]      (row-stochastic, since A1, A2 are)
```

We sweep the Dirichlet concentration `alpha`:

- **small alpha** → peaked/sparse rows → `A_chain` differs sharply from `A2`,
  so composition matters and the single-hop shortcut fails;
- **large alpha** → near-uniform rows → `A_chain ≈ A2 ≈ uniform`, so
  composition is nearly trivial.

### Canonical measurement condition

- `num_heads = 4`, `seq_len = 12`
- `alpha ∈ {0.1, 0.3, 1.0, 3.0, 10.0}` (the sweep axis)
- canonical alpha = `0.3` (peaked enough that composition clearly matters)
- `8` random seeds per alpha, averaged
- evaluation batch uses a fixed seed (`generate(seed=42)`); `generate` is
  deterministic for any given seed.

## Model function signature

The goal's contract with attempts. An attempt provides a `model_fn` and hands
it to `task.evaluate`; it never builds the payload itself.

```python
def model_fn(A1: np.ndarray, A2: np.ndarray) -> np.ndarray:
    """
    Args:
        A1: (num_heads, seq_len, seq_len)  layer-1 row-stochastic attention
        A2: (num_heads, seq_len, seq_len)  layer-2 row-stochastic attention

    Returns:
        A_chain_pred: (num_heads, seq_len, seq_len)  predicted composed attention
    """
```

The returned matrix need not be exactly row-stochastic — `task.evaluate`
projects each row to a probability distribution (clip negatives, renormalise;
all-zero rows become uniform) before scoring, so any finite output is valid.
`task.random_model_fn()` returns a reference `model_fn` emitting a random
row-stochastic matrix of the correct shape (used by the smoke test).

## Payload contract

`task.evaluate(model_fn)` returns a dict with exactly these keys:

```python
{
    "version": 1,                       # int, matches benchmark.VERSION
    "model_name": "synthetic_attention_matrix_chain",
    "num_heads": 4,                     # int
    "seq_len": 12,                      # int
    "canonical_alpha": 0.3,             # float, the canonical condition
    "alpha_sweep": [0.1, 0.3, 1.0, 3.0, 10.0],   # list[float], the sweep axis
    "sweep": [                          # one record per alpha_sweep value
        {
            "alpha": 0.3,              # float, Dirichlet concentration
            "chain_fidelity": 0.91,    # float in [0,1], 1 - mean row TV distance
            "row_kl": 0.12,            # float >= 0, mean KL(true || pred) in nats
            "n_seeds": 8,              # int
        },
        ...
    ],
    "single_hop_baseline": [            # same axis, no-composition reference
        {
            "alpha": 0.3,              # float
            "chain_fidelity": 0.55,    # float in [0,1], fidelity of A2 vs A_chain
            "n_seeds": 8,              # int
        },
        ...
    ],
}
```

`sweep` and `single_hop_baseline` are both lists the same length as
`alpha_sweep`, each indexed by its `alpha` field. `chain_fidelity` is in
`[0, 1]` (bigger = better); `row_kl` is `>= 0` (smaller = better).

## Metrics

`benchmark.score(payload)` returns a flat dict of scalars (floats use `0p3`
form, not `0.3`):

| metric | meaning | direction |
|--------|---------|-----------|
| `version` | `benchmark.VERSION` (= 1) | — |
| `chain_fidelity_alpha_0p1` … `_alpha_10p0` | per-alpha reconstruction fidelity | **bigger = better** |
| `row_kl_alpha_0p1` … `_alpha_10p0` | per-alpha mean row KL | smaller = better |
| `single_hop_baseline_fidelity_alpha_0p1` … `_alpha_10p0` | baseline fidelity per alpha | reference |
| `lift_over_baseline_alpha_0p1` … `_alpha_10p0` | fidelity − baseline, per alpha | bigger = better |
| `chain_fidelity_canonical` | fidelity at `canonical_alpha` (0.3) | **bigger = better** |
| `lift_over_baseline_canonical` | fidelity − baseline at canonical alpha | bigger = better |
| `chain_fidelity_mean` | mean fidelity across the sweep | bigger = better |
| `composition_robustness` | fidelity at most-peaked (alpha 0.1) ÷ fidelity at most-uniform (alpha 10.0), clipped `[0,1]` | **bigger = better** (headline) |

### Headline summary

**`composition_robustness`** — the fraction of the easy-case (near-uniform,
`alpha = 10.0`) fidelity that survives in the hard case (peaked, `alpha = 0.1`)
where the single-hop shortcut collapses. A mechanism that genuinely computes
the matrix product degrades gracefully and scores near `1.0`; one that leans on
a single-hop approximation scores low.

## Pipeline hooks

- `GPU_REQUIREMENT = 1` — attempts run on the GPU (the smoke test runs
  `task`/`benchmark` on CPU/NumPy).
- `is_obviously_broken(metrics)` — short-circuits the jury when metrics are
  NaN/inf or fail to beat the single-hop baseline at the canonical condition
  (`chain_fidelity_canonical <= single_hop_baseline_fidelity_alpha_0p3`).

## Bump procedure

Bump `VERSION` (in `benchmark.py` and this README, same commit) when:

- any metric formula changes;
- a payload key is added/removed/renamed or retyped;
- `canonical_alpha` or the sweep values change;
- a sweep record's schema changes.

Do **not** bump when adding a new metric that leaves existing ones unchanged,
or adding an optional payload key with a default. This goal is at `VERSION = 1`.
Old `benchmark.json` files stay on disk; the dashboard filters to the highest
version present.
