# attention_matmul

## Question

Can a mechanistic explanation identify *which query-key interactions* drive an
attention head's output? Given Q, K, V, the attention computation is
`softmax(QK^T/√d) @ V`. This goal asks: can an explanation method produce an
**attribution matrix** `Attrib[b, h, i, j]` — the estimated contribution of key
`j` to the output at query `i` — that matches the true computational pathway,
and does that fidelity hold as queries and keys become more aligned (cosine
grows) or fully unstructured?

## Setup

**Synthetic generator** — fully controlled, no trained models. We construct
Q, K, V where the ground-truth attention `A = softmax(QK^T/√d_head)` has a
known structure, and the output `O = A @ V` is therefore a known linear
combination of the (random Gaussian) value vectors.

### Canonical measurement condition

- `d_model = 64`, `n_heads = 4`, `d_head = 16`, `seq_len = 32`,
  `batch_size = 16`.
- Sweep axis `qk_alignment` over four regimes:

  | condition    | description                                                   | ground-truth attribution        |
  |--------------|---------------------------------------------------------------|----------------------------------|
  | `orthogonal` | each query equals exactly one key (orthonormal basis)         | one-hot per query                |
  | `cos_0p3`    | each query has cosine ≈ 0.3 with one designated key           | dominant key + diffuse tail      |
  | `cos_0p7`    | each query has cosine ≈ 0.7 with one designated key           | strongly peaked                  |
  | `uniform`    | queries and keys i.i.d. Gaussian                              | diffuse, no dominant key         |

- **Canonical headline condition:** `cos_0p3`.
- Evaluation uses a fixed seed (`generate(seed=0)`); `generate` is deterministic
  for any given seed.
- No fine-tuning, no gradients — pure inference.

## Model function signature

The goal's contract with attempts. An attempt provides a `model_fn` and hands
it to `task.evaluate`; it never builds the payload itself.

```python
def model_fn(
    Q: np.ndarray,   # (batch, n_heads, seq_len, d_head)
    K: np.ndarray,   # (batch, n_heads, seq_len, d_head)
    V: np.ndarray,   # (batch, n_heads, seq_len, d_head)
) -> np.ndarray:     # (batch, n_heads, seq_len, seq_len) — attribution matrix
    ...
```

`Attrib[b, h, i, j]` is the estimated contribution of key `j` to the output at
query position `i`. Rows should approximately sum to 1 (each output is a convex
combination of values); entries are ideally non-negative. `task.evaluate`
applies the reductions and computes all metrics. `task.random_model_fn()`
returns a reference `model_fn` that emits uniform attribution of the correct
shape (used by the smoke test).

## Payload contract

`task.evaluate(model_fn)` returns a dict with exactly these keys. **It holds
only scalars and small dicts — no raw tensors.**

```python
{
    "version": 1,                              # int, matches benchmark.VERSION
    "model_name": "synthetic_attention_matmul",
    "config": {                                # frozen generation config
        "d_model": 64, "n_heads": 4, "d_head": 16,
        "seq_len": 32, "batch_size": 16,
    },
    "canonical_condition": "cos_0p3",          # str, the headline condition
    "conditions": ["orthogonal", "cos_0p3", "cos_0p7", "uniform"],
    "sweep": [                                 # one record per condition
        {
            "qk_alignment": "orthogonal",      # str, condition name
            "output_mse": 0.12,                # float, MSE((Attrib @ V), true O)
            "attribution_kl": 0.34,            # float, mean KL(true_attn || Attrib)
            "rowsum_mae": 0.01,                # float, mean |rowsum(Attrib) - 1|
        },
        ...                                    # one per condition
    ],
    "linear_baseline": {                       # fixed strawman: uniform attribution
        "orthogonal": {"output_mse": ..., "attribution_kl": ..., "rowsum_mae": ...},
        "cos_0p3":    {"output_mse": ..., "attribution_kl": ..., "rowsum_mae": ...},
        "cos_0p7":    {"output_mse": ..., "attribution_kl": ..., "rowsum_mae": ...},
        "uniform":    {"output_mse": ..., "attribution_kl": ..., "rowsum_mae": ...},
    },
}
```

`sweep` is indexed by its `qk_alignment` field; `linear_baseline` is keyed by
the same condition names. Lower `output_mse`, `attribution_kl`, and
`rowsum_mae` are better.

## Metrics

`benchmark.score(payload)` returns a flat dict of scalars. Fidelity metrics are
**normalised against the uniform baseline**: `1 - value / value_baseline`,
clipped to `[0, 1]`, so `0.0` = no better than baseline and `1.0` = perfect.

| metric | formula | direction |
|--------|---------|-----------|
| `version` | `benchmark.VERSION` (= 1) | — |
| `attribution_fidelity_canonical` | fidelity of `attribution_kl` at `cos_0p3` | **bigger = better** (headline) |
| `attribution_fidelity_qk_<cond>` | per-condition KL fidelity (`orthogonal`, `cos_0p3`, `cos_0p7`, `uniform`) | bigger = better |
| `output_reconstruction_qk_<cond>` | per-condition `output_mse` fidelity | bigger = better |
| `output_reconstruction_canonical` | output reconstruction at `cos_0p3` | bigger = better |
| `rowsum_mae_qk_<cond>` | raw mean `|rowsum - 1|` per condition | smaller = better |
| `attribution_fidelity_mean` | mean of per-condition attribution fidelity | bigger = better |
| `linear_baseline_attribution_fidelity_qk_<cond>` | `0.0` by definition | reference |
| `linear_baseline_output_reconstruction_qk_<cond>` | `0.0` by definition | reference |

### Headline summary

**`attribution_fidelity_canonical`** — the fraction of the uniform baseline's
attention KL that the method removes at the canonical `cos_0p3` regime. A method
that recovers the true query-key pathway scores near `1.0`; one no better than
uniform attribution scores `0.0`.

### Edge cases

- A (near-)zero baseline error means the baseline is already perfect, so the
  normalisation is ill-defined — fidelity is set to `1.0`.
- A method *worse* than the baseline yields a negative raw fidelity, clipped to
  `0.0`.
- Non-finite KL/MSE (NaN/inf) maps to fidelity `0.0`; `attribution_kl` inputs
  are clipped at `1e6` before normalisation.

## Pipeline hooks

- `GPU_REQUIREMENT = 1` — attempts run on the GPU (the smoke test runs
  `task`/`benchmark` on CPU/NumPy).
- `is_obviously_broken(metrics)` — short-circuits the jury when any metric is
  NaN/inf, or when `attribution_fidelity_canonical <= 0.0` (no better than the
  uniform baseline at the canonical condition).

## Bump procedure

Bump `VERSION` (in `benchmark.py` and this README, same commit) when:

- any metric formula changes;
- a payload key is added/removed/renamed or retyped;
- the canonical condition or the sweep conditions change;
- a sweep / baseline record's schema changes.

Do **not** bump when adding a new metric that leaves existing ones unchanged,
or adding an optional payload key with a default. This goal is at `VERSION = 1`.
