# Attention Regex: Mechanistic Interpretability Goal

## Question
Can an attention mechanism implement **regex-like sequence pattern matching** —
concentrating attention on the positions where a (wildcard-capable) token
pattern *finishes matching* a sequence, rather than merely firing on the
pattern's final token? A single attention head can spot a 1-gram by content
alone; matching a multi-token pattern requires composing information across the
preceding context. We measure how sharply attention separates true match-end
positions from the rest, and how that sharpness holds up as the pattern (and so
the required composition depth) grows.

## Setup
**Synthetic generator only.** No trained model required. The generator produces
a token embedding matrix, a wildcard-capable pattern, and an embedded token
sequence whose ground-truth match-end positions are known. The attempt provides
a `model_fn` that, given the pattern, the embeddings, and the residual stream,
emits per-position logits; we softmax them and score how well attention lands on
the true match-end positions.

### Pattern family
- Alphabet of `VOCAB_SIZE = 8` tokens, embedded in `d = 64` dimensions
  (unit-norm rows, resampled per entry).
- A pattern is a length-`L` array of token ids in `[0, VOCAB_SIZE)`, where each
  position is independently a **wildcard** (`-1`, matches any token) with
  probability `0.25`. At least one concrete token is always kept.
- A **match ends** at position `i` if `sequence[i-L+1 : i+1]` matches the
  pattern (wildcards match anything). `N_PLANTS = 6` matches are planted at
  distinct start positions per sequence so positives always exist; incidental
  matches in the random background are also labelled.
- Sequence length `N_POSITIONS = 120`. The residual stream is
  `embed[token] * 2.0 + noise(0.5)`.

### Canonical measurement condition
- Pattern length `L = 3` (`canonical_length`).
- `VOCAB_SIZE = 8`, `d = 64`, `N_POSITIONS = 120`.
- Length sweep `L ∈ {1, 2, 3, 4, 5, 6}`, `N_SEEDS = 10` per length.
- Evaluation seed `EVAL_SEED = 42` (the batch `generate` produces is fixed).

## Model function signature
```python
def model_fn(pattern: np.ndarray, embed: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """
    Args:
        pattern:  int array of shape (L,); token ids in [0, VOCAB_SIZE),
                  with -1 marking a wildcard.
        embed:    float array of shape (VOCAB_SIZE, d); row v is token v's embedding.
        residual: float array of shape (N, d); the embedded token sequence + noise.
    Returns:
        logits:   float array of shape (N,); per-position attention logits.
                  These are softmaxed downstream — only relative values matter.
    """
```
The attempt's `main.py` implements this. It may build any mechanism (attention,
convolution, learned weights) but must return exactly one logit per position.

## Payload contract
`task.evaluate(model_fn)` returns a dict with these exact keys:

```python
{
    "version": 1,                       # payload schema version
    "model_name": "synthetic_attention_regex",
    "d": 64,
    "vocab_size": 8,
    "n_positions": 120,
    "canonical_length": 3,
    "length_sweep": [1, 2, 3, 4, 5, 6],
    "sweep": [                          # one record per length in length_sweep
        {
            "length": 1,
            "match_sharpness": float,       # [0,1] separation of attn on match vs non-match
            "false_positive_rate": float,   # [0,1] fp / (fp + tn)
            "false_negative_rate": float,   # [0,1] fn / (fn + tp)
            "n_seeds": 10,
        },
        # ... one record for each of L = 2..6
    ],
    "linear_baseline": [                # one record per length, same order
        {"length": 1, "match_sharpness": float, "n_seeds": 10},
        # ... L = 2..6
    ],
}
```

### Payload field semantics
| Key | Meaning |
|-----|---------|
| `match_sharpness` | Mean over seeds of `(mean attn on match-end positions − mean attn elsewhere) / max(|mean on|, 1e-8)`, clipped to `[0, 1]`. Higher = attention concentrates on matches. |
| `false_positive_rate` | Positions predicted (attn above the uniform level `1/N`) that are not match-ends, over all non-match positions. |
| `false_negative_rate` | Match-end positions not predicted, over all match-end positions. |
| `linear_baseline.match_sharpness` | Same sharpness for the no-composition strawman: score each position by similarity to the embedding of the pattern's **last concrete token** only (ignoring preceding context). |

`match_sharpness` and both rates are aggregated by averaging over `N_SEEDS` seeds.

## Metrics
`benchmark.score(payload)` returns a flat dict (floats unless noted):

| Metric | Formula | Direction |
|--------|---------|-----------|
| `version` | `VERSION` (first key) | — |
| `match_sharpness_len_<L>` | sweep record's `match_sharpness` | **bigger better** |
| `false_positive_rate_len_<L>` | sweep record's `false_positive_rate` | smaller better |
| `false_negative_rate_len_<L>` | sweep record's `false_negative_rate` | smaller better |
| `linear_baseline_sharpness_len_<L>` | baseline record's `match_sharpness` | reference |
| `match_sharpness_canonical` | `match_sharpness_len_3` | **bigger better** |
| `lift_over_baseline_canonical` | `match_sharpness_canonical − linear_baseline_sharpness_len_3` | bigger better |
| `length_robustness` | `sharpness(L=6) / sharpness(L=1)`, clipped to `[0,1]` | **bigger better (headline)** |

### Headline summary metric
**`length_robustness`** — the fraction of single-token sharpness retained at the
longest (6-token) pattern. A mechanism that only spots the final token collapses
toward 0 as `L` grows; a genuine multi-token matcher stays near 1. This is the
single number on the leaderboard.

### Baseline
The **linear baseline** scores each position purely by its similarity to the
pattern's last concrete token's embedding — no sequential composition. It is
computed under identical conditions and reported as `linear_baseline_sharpness_len_<L>`.
A method beating this baseline (positive `lift_over_baseline_canonical`) is the
meaningful signal; matching it is not.

### Edge cases
- Empty/zero denominators in the rates use `max(denom, 1)`; sharpness with no
  positives or no negatives is defined as `0.0`.
- `length_robustness` is `0.0` when the shortest-length sharpness is `≤ 1e-12`.
- `is_obviously_broken` flags any NaN/inf metric, or a canonical sharpness that
  fails to beat the canonical (`L=3`) linear baseline — skipping the jury.

## Pipeline hooks
- `GPU_REQUIREMENT = 1` — attempts run on the GPU; `task.py`/`benchmark.py`
  stay pure CPU/NumPy.
- `is_obviously_broken(metrics)` — short-circuits the jury on mechanically
  degenerate results only.

## Bump procedure
Increment `VERSION` in `benchmark.py` (and update this contract in the same
commit) when:
- payload keys are added/removed/renamed/retyped;
- any existing metric formula changes;
- the canonical condition changes (`canonical_length`, `d`, vocab, `N_POSITIONS`).

Adding a new length to `length_sweep` or a new metric without touching existing
ones does **not** require a bump.
