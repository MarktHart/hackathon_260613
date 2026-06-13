# attention_viterbi

## Question

Does a transformer trained on sequences from a Hidden Markov Model (HMM)
implement the **Viterbi recurrence** in its attention — i.e., does some head
dynamically concentrate attention on the position that the Viterbi backpointer
identifies as the predecessor of the current state?

For a first-order HMM the Viterbi backpointer of query position `t` is always
`t-1` (the chain has no longer-range dependency), so the mechanistic signature
we look for is a head that places **more attention mass on position `t-1`** than
a uniform reader of the causal past would. This is the attention substrate the
Viterbi dynamic program needs.

## Setup

- **Synthetic generator.** A fixed 3-state, 4-observation HMM with known
  parameters (frozen in `task.py`):
  - States `S = {0, 1, 2}`, observations `O = {0, 1, 2, 3}`.
  - Initial distribution `π = [0.6, 0.3, 0.1]`.
  - Transition `A` (row-stochastic):
    ```
    [[0.7, 0.2, 0.1],
     [0.1, 0.8, 0.1],
     [0.2, 0.3, 0.5]]
    ```
  - Emission `B` (row-stochastic):
    ```
    [[0.80, 0.10, 0.05, 0.05],
     [0.10, 0.70, 0.10, 0.10],
     [0.05, 0.15, 0.70, 0.10]]
    ```
- **Model.** A 2-layer, 4-head, `d_model=64` causal transformer
  (attention-only) trained on next-token prediction over HMM sequences. The
  attempt owns training; the goal only specifies the evaluation interface.
- **Canonical measurement condition.** The **same 100 sequences of length 20**
  produced by `task.generate(seed=42)`. Every attempt is scored on exactly
  these sequences. Other seeds are for ablation only.

## Model function signature

The attempt's `main.py` must expose a single callable named `model_fn`:

```python
def model_fn(input_ids: np.ndarray) -> dict[str, np.ndarray]:
    """
    Args:
        input_ids: int array, shape [batch, seq_len], token IDs in 0..3.

    Returns:
        {
          "attn_weights": float array [batch, n_layers, n_heads, seq_len, seq_len],
              # causal, post-softmax attention. Layer 0 is the bottom layer.
              # Row t must be a distribution over keys 0..t (causal mask).
          "logits": float array [batch, seq_len, vocab_size],   # vocab_size = 4
        }
    """
```

Canonical shapes: `n_layers=2`, `n_heads=4`, `seq_len=20`, `vocab_size=4`,
`batch=100`. `evaluate` raises `ValueError` on any shape mismatch.

## Payload contract

`task.evaluate(model_fn)` returns exactly:

```python
{
    "version": 1,                       # matches benchmark.VERSION
    "model_config": {...},              # n_layers, n_heads, d_model, seq_len, vocab_size
    "hmm_config": {...},                # n_states, n_obs, pi, A, B
    "n_layers": 2,
    "n_heads": 4,
    "seq_len": 20,
    "eval_sequences": [[...], ...],     # 100 int lists, len 20 (self-describing)
    "viterbi_paths": [[...], ...],      # 100 int lists, len 20 (states 0..2)
    "best_head": {"layer": int, "head": int},
    "per_head": [                       # one record per layer×head (8 total)
        {"layer": 0, "head": 0, "excess": 0.41},
        ...
    ],
    "positional": [                     # best head, one record per query pos 1..19
        {"pos": 1, "excess": 0.12, "n": 100},
        ...
    ],
    "baseline_uniform_excess": 0.0,     # excess under uniform causal attention
    "baseline_random_excess": -0.001,   # excess under random Dirichlet attention
}
```

### How `excess` is computed (per head)

For each query position `t = 1..T-1` and each sequence, the **excess attention
on the Viterbi predecessor** is

```
excess(t) = α[t, t-1] − mean(α[t, 0:t])
```

where `α` is the head's causal attention matrix. `excess` for a head is the mean
of `excess(t)` over all `(sequence, t)` pairs. Properties:

- Bounded in `(-1, 1)`. `α[t, t-1] ∈ [0, 1]` and `mean(α[t, 0:t]) = 1/t` for a
  normalised causal row.
- **Exactly 0** for uniform causal attention — the no-mechanism reference.
- Positive when the head concentrates on the predecessor (the Viterbi signature).

`positional` reports `excess(t)` for the **single strongest head** so the
dashboard can show where in the sequence the structure emerges.

## Metrics

| Metric | Formula / Description | Bigger is better? |
|--------|----------------------|-------------------|
| `viterbi_attention_canonical` | **Headline.** Max `excess` over all 8 heads — the strongest Viterbi head. | Yes |
| `viterbi_attention_mean` | Mean `excess` over all 8 heads. | Yes |
| `viterbi_attention_layer_<l>_head_<h>` | Per-head `excess` (8 metrics). | Yes |
| `viterbi_attention_pos_<p>` | Best head's `excess` at query position `p` (1..19). | Yes |
| `viterbi_robustness` | Fraction of query positions where the best head's `excess > 0`. In `[0, 1]`. | Yes |
| `linear_baseline_viterbi_attention` | `excess` of the uniform causal-attention baseline (≡ 0). Reference. | N/A |
| `baseline_uniform_excess` / `baseline_random_excess` | No-mechanism references under identical conditions. | N/A |
| `lift_over_uniform` | `viterbi_attention_canonical − baseline_uniform_excess`. | Yes |
| `lift_over_random` | `viterbi_attention_canonical − baseline_random_excess`. | Yes |
| `best_head_layer` / `best_head_head` | Index of the strongest head (informational). | N/A |

All `excess`/`viterbi_attention_*` metrics share one direction (bigger is
better). The uniform baseline is `0`, so any positive canonical value beats it.

`is_obviously_broken` returns `True` (skipping the jury) when any metric is
NaN/inf, or when `viterbi_attention_canonical ≤ baseline_uniform_excess` —
i.e. no head beats a uniform reader, so there is no mechanism to judge.

## Bump procedure

Bump `VERSION` in `benchmark.py` **and** `version` in the payload together when:

- the HMM parameters, sequence length, count, or canonical seed (42) change;
- the `excess` formula changes;
- any payload key is added-as-required, removed, renamed, or retyped.

You do **not** bump when adding a new optional metric or an optional payload key
with a default. Update this contract table in the same commit as any bump.
