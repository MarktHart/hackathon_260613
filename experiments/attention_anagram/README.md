# attention_anagram

## Question
Do attention heads in transformer models learn to explicitly represent token permutations when processing anagram pairs? Specifically, when a target sequence is an anagram (permutation) of a source sequence, does the attention from target positions to source positions concentrate on the correct source position for each target token?

## Setup
**Synthetic generator** — fully controlled, no trained model required for the canonical evaluation. The generator produces pairs of sequences `(source, target)` where `target` is a permutation of `source`. The ground-truth permutation is known.

We evaluate on a **single-layer, multi-head attention-only model** (or extract a single layer from a larger model). The model function receives source and target token IDs and returns attention weights from each target position to each source position.

## Canonical measurement condition
- Sequence length: 8 tokens
- Vocabulary size: 50
- Batch size: 500 examples **total**, drawn with a per-example permutation type
  chosen uniformly at random from `{swap, rotation, random}`. Each slice
  (`swap`/`rotation`/`random`) is therefore computed over roughly 1/3 of the
  500 examples; the exact split is fixed by `seed=0`.
- Headline permutation type: **random permutation** (uniform over all 8!
  permutations) — the `anagram_alignment_canonical` metric is the `random` slice.
- Model: 8 heads, 1 layer. The canonical `model_fn` contract returns a single
  layer's attention `(batch, n_heads, tgt_len, src_len)`; every sweep record is
  emitted at `layer_idx=0`. To evaluate a multi-layer model per-layer, call
  `evaluate` once per extracted layer (the `score` function keeps only
  `layer_idx == 0` records).

The headline uses random permutations because they are maximally unstructured and thus require the model to learn the specific mapping rather than exploit simple patterns (like adjacent swaps or rotations).

## Model function signature
```python
def model_fn(src_ids: np.ndarray, tgt_ids: np.ndarray) -> np.ndarray:
    """
    Args:
        src_ids: int array of shape (batch, src_len), token IDs of source sequences.
        tgt_ids: int array of shape (batch, tgt_len), token IDs of target sequences.
                 tgt_len == src_len always holds.
    Returns:
        attn: float array of shape (batch, n_heads, tgt_len, src_len).
              Attention weights from each target position to each source position.
              For each (batch, head, tgt_pos), the weights over src_pos should sum to 1.
    """
```

## Payload contract
`task.evaluate(model_fn)` returns a dict with the following keys:

```python
{
    "version": 1,                                    # int, payload schema version
    "config": {                                      # dict, frozen generation config
        "seq_len": 8,
        "vocab_size": 50,
        "batch_size": 500,
        "perm_types": ["swap", "rotation", "random"],
        "n_heads": 8,
        "n_layers": 1,
    },
    "sweep": [                                       # list of per-condition records
        {
            "perm_type": "swap" | "rotation" | "random",
            "head_alignments": [                     # length n_heads
                {
                    "head_idx": 0,
                    "mean_alignment": float,         # mean over batch & positions of attention on true source pos
                    "max_alignment": float,          # max over positions
                    "alignment_per_pos": [float],    # length seq_len, mean over batch
                },
                ...
            ],
            "layer_idx": 0,
        },
        ...
    ],
    "random_baseline": {                             # same structure, computed from uniform attention
        "swap": {"mean_alignment": 1/8, ...},
        "rotation": {"mean_alignment": 1/8, ...},
        "random": {"mean_alignment": 1/8, ...},
    },
}
```

**Semantics**:
- `mean_alignment`: For a given head and permutation type, average over all examples and all target positions of the attention weight placed on the *true* source position for that target token. Range `[0, 1]`. Random attention yields `1/seq_len = 0.125`.
- `max_alignment`: Maximum over target positions of the per-position mean alignment.
- `alignment_per_pos`: List of length `seq_len`; each entry is the mean alignment at that target position (averaged over batch).
- `random_baseline`: The expected alignment under uniform attention (`1/seq_len` for all). Included so `benchmark.score` can compute lift without recomputing.

## Metrics
Returned by `benchmark.score(payload)` — a flat dict of scalars.

| Metric | Formula | Direction | Notes |
|--------|---------|-----------|-------|
| `version` | `payload["version"]` | — | Always first key. |
| `anagram_alignment_canonical` | Mean of `mean_alignment` across heads for `perm_type="random"` at `layer_idx=0`. | **Bigger is better** | Headline summary. |
| `anagram_alignment_swap` | Same, for `perm_type="swap"`. | **Bigger is better** | Per-slice. |
| `anagram_alignment_rotation` | Same, for `perm_type="rotation"`. | **Bigger is better** | Per-slice. |
| `anagram_alignment_random` | Same as canonical; explicit per-slice name. | **Bigger is better** | Per-slice. |
| `anagram_alignment_max_canonical` | Max of `max_alignment` across heads for `perm_type="random"`. | **Bigger is better** | Best head, not average. |
| `alignment_robustness` | `min(anagram_alignment_swap, anagram_alignment_rotation, anagram_alignment_random) / max(...)` | **Bigger is better** | In `[0, 1]`. Measures consistency across permutation structures. |
| `lift_over_random_canonical` | `anagram_alignment_canonical - 1/seq_len` | **Bigger is better** | Excess over uniform baseline. |
| `random_baseline_alignment` | `1/seq_len` (0.125) | — | Reference constant. |

## Bump procedure
- Bump `VERSION` in `benchmark.py` and update this README when:
  - Any metric formula changes.
  - Payload keys are added, removed, or retyped.
  - Canonical condition changes (seq_len, vocab_size, batch_size, perm_type).
- Adding a new `perm_type` to the sweep does **not** require a bump (extensible list).
- Adding a new metric without changing existing ones does **not** require a bump.