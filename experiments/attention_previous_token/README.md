# attention_previous_token

## Question

Can a single attention head implement a clean **previous-token** pattern —
attending, from each query position `i`, sharply to the key at position
`i - 1`? This is the canonical induction-head *precursor*: before a model can
do in-context copying it needs a head that reliably shifts attention back by
exactly one position, using positional information while ignoring token
content. We also ask how robust that pattern is as the residual stream is
corrupted with noise.

## Setup

**Synthetic generator** — fully controlled, no trained models. We build a
residual stream of length `seq_len` where each position carries a sinusoidal
**positional embedding** plus random per-position **content**. A genuine
previous-token head must read position (not content) so that query `i` aligns
with key `i - 1`.

The attempt supplies a `model_fn` (its "head") that maps the residual stream
to a matrix of attention logits. `task.evaluate` applies a **causal mask** and
row-wise softmax, then measures how much attention mass lands on the previous
token. We sweep an additive Gaussian **noise** level on the residual to probe
robustness; a perfect head keeps attending to `i - 1` as noise grows.

### Canonical measurement condition

- `seq_len = 64`
- `d = 64` (residual dimension)
- `n_seeds = 16` independent sequences, averaged
- content scale = 0.5, positional scale = 1.0
- canonical noise = `0.0`
- noise sweep: `[0.0, 0.25, 0.5, 1.0, 2.0]`
- data seed fixed at `0`; the additive-noise RNG is fixed inside `evaluate`,
  so every attempt sees identical data.

## Model function signature

```python
def model_fn(residual: np.ndarray) -> np.ndarray:
    """
    Args:
        residual: (seq_len, d) float32 — residual stream for one sequence.

    Returns:
        attn_logits: (seq_len, seq_len) float32 — unnormalised attention
                     logits. Entry [i, j] is the logit for query i attending
                     to key j. The future (j > i) is masked out by the
                     evaluator, so values there are ignored; higher = more
                     attention.
    """
```

The attempt returns raw logits; `task.evaluate` masks the future, softmaxes
each row, and computes the metrics. Pure NumPy in, pure NumPy out.

## Payload contract

`task.evaluate` returns a dict with exactly these keys:

```python
{
    "version": 1,                                # matches benchmark.VERSION
    "model_name": "synthetic_previous_token",    # fixed identifier
    "seq_len": 64,
    "d": 64,
    "canonical_noise": 0.0,                       # the canonical condition
    "noise_sweep": [0.0, 0.25, 0.5, 1.0, 2.0],    # the sweep axis values
    "uniform_baseline": 0.0594,                   # mean over i>=1 of 1/(i+1)
    "sweep": [                                    # one record per noise value
        {
            "noise": 0.0,
            "prev_token_attention": 0.91,         # mean A[i, i-1] over i>=1
            "self_attention": 0.02,               # mean A[i, i]   (distractor)
            "two_back_attention": 0.01,           # mean A[i, i-2] (distractor)
            "uniform_baseline": 0.0594,           # same reference, per record
            "n_seeds": 16,
        },
        ...
    ],
}
```

All attention values are in `[0, 1]` (rows of a softmax). `uniform_baseline`
is the previous-token mass a content-blind uniform causal head would achieve:
`mean over i in [1, seq_len-1] of 1/(i+1)`. It is the no-mechanism reference.

## Metrics

`benchmark.score(payload)` returns a flat dict:

| metric | formula | direction |
|--------|---------|-----------|
| `version` | `payload["version"]` | — |
| `prev_token_attn_canonical` | prev-token mass at canonical noise (0.0) | **bigger = better (headline)** |
| `prev_token_attn_noise_0p00` .. `prev_token_attn_noise_2p00` | per-noise prev-token mass | bigger = better |
| `self_attn_canonical` | self-attention mass at canonical noise | smaller = better |
| `two_back_attn_canonical` | two-back mass at canonical noise | smaller = better |
| `uniform_baseline` | `mean_{i>=1} 1/(i+1)` | reference |
| `lift_over_uniform_canonical` | `prev_token_attn_canonical - uniform_baseline` | bigger = better |
| `prev_token_lift_ratio_canonical` | `prev_token_attn_canonical / uniform_baseline` | bigger = better |
| `prev_token_robustness` | `clip(prev@max_noise / prev@0, 0, 1)` | bigger = better |

### Headline summary

**`prev_token_attn_canonical`** — the single number to optimise: the fraction
of attention mass a head places on the previous token under clean (noise = 0)
conditions. A perfect previous-token head approaches `1.0`; a content-blind
uniform head sits at `uniform_baseline`.

`prev_token_robustness` is the secondary story: a head that holds its
previous-token pattern under the heaviest noise scores near `1.0`; one that
collapses to uniform scores near `0.0`.

## Bump procedure

Bump `VERSION` (in `benchmark.py` and this README) when:
- any metric formula changes;
- payload keys are added/removed/renamed, or a sweep-record field changes;
- the canonical noise, the noise sweep, or `seq_len`/`d` change.

Do **not** bump when adding a new metric that leaves existing ones untouched,
or when adding an optional payload key with a default.
