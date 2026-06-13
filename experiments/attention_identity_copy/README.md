# Attention Identity Copy

## Question

Does the model contain an attention head that implements the **identity copy** operation — i.e., for a given position, the attention output equals the value vector at that same position (copying the token representation faithfully)?

This is a fundamental building block: many circuits (induction, copying, transfer) rely on heads that can route information from position *i* to position *i* without distortion.

## Setup

**Synthetic generator.** No trained model is required. We construct a minimal "model function" that takes a batch of token sequences and returns per-head attention weights and value projections. The task evaluates whether any head behaves as an identity copier across a sweep of token values.

- **Sequence length**: 16
- **Vocabulary size**: 256 (tokens 0–255)
- **Model function signature** (the contract between goal and attempts):

```python
def model_fn(batch: Batch) -> ModelOutput:
    ...
@dataclass(frozen=True)
class Batch:
    tokens: np.ndarray          # shape (B, L), int32, values in [0, 255]
    # no other fields — the generator is the sole source of randomness

@dataclass(frozen=True)
class ModelOutput:
    attn_weights: np.ndarray    # shape (B, H, L, L), float32, rows sum to 1
    values: np.ndarray          # shape (B, H, L, D), float32
    # D = head dimension, H = number of heads
```

The attempt's `main.py` implements `model_fn` however it likes (extract from real model, learn a probe, hand-craft a circuit). The goal only cares about the *output* tensors.

## Canonical measurement condition

- **Batch size** `B = 32`
- **Sequence length** `L = 16`
- **Heads** `H = 8`
- **Head dimension** `D = 64`
- **Sweep tokens**: `[0, 64, 128, 192, 255]` (5 tokens spanning the vocab)
- For each sweep token `t`, we construct a batch where **every position in every sequence is token `t`**. The identity-copy head should attend uniformly to the diagonal (or a fixed offset) and copy the value vector faithfully.

## Payload contract

`task.evaluate(model_fn)` returns a `dict` with exactly these keys:

```python
{
    "version": 1,                              # payload schema version
    "sweep": [
        {
            "token": int,                      # the sweep token value
            "copy_fidelity": float,            # cosine similarity between attention output and value at the same position, for the BEST head (max over heads), averaged over batch and positions
            "diag_attn_mass": float,           # attention mass on the diagonal (position i → i) for that same best head, averaged over batch and positions
            "best_head": int,                  # head index (0..H-1) achieving max copy_fidelity for this token
        }
        for token in [0, 64, 128, 192, 255]
    ],
    "canonical_token": 128,                    # the token used for headline metric
    "config": {
        "B": 32, "L": 16, "H": 8, "D": 64,
        "sweep_tokens": [0, 64, 128, 192, 255],
    },
}
```

All floats are Python `float` (not numpy scalars). `copy_fidelity ∈ [-1, 1]`, `diag_attn_mass ∈ [0, 1]`.

## Metrics

`benchmark.score(payload)` returns a flat `dict[str, float | int]`:

| Metric | Formula | Direction | Meaning |
|--------|---------|-----------|---------|
| `version` | `payload["version"]` | — | Benchmark version |
| `identity_copy_fidelity_canonical` | `copy_fidelity` at `canonical_token` (128) | **Bigger is better** | Headline: how well the best head copies at the mid-vocab token |
| `identity_copy_fidelity_token_<t>` | `copy_fidelity` for each sweep token `t` | **Bigger is better** | Per-token slice (5 values: `token_0`, `token_64`, `token_128`, `token_192`, `token_255`) |
| `diag_attn_mass_token_<t>` | `diag_attn_mass` for each sweep token | **Bigger is better** | Does the head attend to the diagonal? |
| `linear_baseline_fidelity_canonical` | `1 / sqrt(L)` | **Smaller is better** | Baseline: expected fidelity of a uniform-attention (no-mechanism) head, ≈ 0.25 for L=16 |
| `lift_over_linear_baseline` | `identity_copy_fidelity_canonical - linear_baseline_fidelity_canonical` | **Bigger is better** | Improvement over random attention |

All per-token metrics use the token value in the key (e.g., `identity_copy_fidelity_token_128`).

## Bump procedure

- Increment `VERSION` in `benchmark.py` when any metric formula changes, payload keys are added/removed/retyped, or the canonical token/sweep changes.
- Update this README's payload contract and metrics table in the same commit.