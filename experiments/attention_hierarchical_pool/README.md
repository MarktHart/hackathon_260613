# Attention Hierarchical Pool Goal

## Question

Do attention heads in a transformer implement **hierarchical pooling** — where early-layer heads attend to local token neighborhoods (fine-grained pooling) and later-layer heads attend to compressed representations of larger chunks (coarse-grained pooling)?

This is a synthetic benchmark: we generate sequences with known hierarchical token groupings and measure whether attention patterns at each layer align with the ground-truth pooling structure.

## Setup

- **Synthetic generator only** — no trained model required. The task constructs a fixed "hierarchical tokenizer" that maps token IDs to positions in a 2-level tree (chunks → sequences).
- **Canonical measurement condition**:
  - Sequence length: 256 tokens
  - 16 chunks of 16 tokens each
  - 4 hierarchy levels (token → chunk → super-chunk → sequence) — we measure the bottom two transitions
  - 12 layers, 8 heads per layer (96 heads total)
  - Deterministic seed = 0
- **Model function signature** (what attempts must implement):

```python
def model_fn(
    input_ids: np.ndarray,          # (batch, seq_len) int32
    layer_idx: int,                 # 0 .. 11
    head_idx: int                   # 0 .. 7
) -> np.ndarray:                    # (batch, seq_len, seq_len) float32 — attention weights, rows sum to 1
    ...
```

The function returns attention weights for **one specific head**. The attempt's `main.py` loops over all (layer, head) pairs and calls `model_fn` for each.

## Payload Contract

`task.evaluate` returns a dict with exactly these keys:

```python
{
    "version": 1,                                    # payload schema version
    "seq_len": 256,
    "num_layers": 12,
    "num_heads": 8,
    "chunk_size": 16,
    "num_chunks": 16,
    "sweep": [                                       # one record per (layer, head)
        {
            "layer": int,                            # 0 .. 11
            "head": int,                             # 0 .. 7
            "local_concentration": float,            # mass on within-chunk positions [0, 1]
            "chunk_concentration": float,            # mass on same-chunk (including local) [0, 1]
            "superchunk_concentration": float,       # mass on same-superchunk (4 chunks) [0, 1]
            "entropy": float                         # attention entropy (nats)
        },
        ...  # 96 records total
    ]
}
```

All concentrations are averages over all query positions and the batch (batch size = 1).

## Metrics

`benchmark.score` returns a flat dict:

| Metric | Formula | Direction |
|--------|---------|-----------|
| `hierarchical_robustness_canonical` | `median_{layer≥6}(spread_L) / median_{layer<6}(spread_L)`, where `spread_L = chunk_concentration_layer_<L> / local_concentration_layer_<L>` | bigger = better |
| `local_concentration_layer_<L>` | median `local_concentration` across heads at layer L | bigger = better for early layers |
| `chunk_concentration_layer_<L>` | median `chunk_concentration` across heads at layer L | bigger = better for mid/late layers |
| `superchunk_concentration_layer_<L>` | median `superchunk_concentration` across heads at layer L | bigger = better for late layers |
| `entropy_layer_<L>` | median `entropy` across heads at layer L | smaller = better (sharper) |
| `linear_baseline_local_concentration_layer_<L>` | baseline: uniform-within-chunk local mass = `74/256 ≈ 0.289` | — |
| `linear_baseline_chunk_concentration_layer_<L>` | baseline: uniform-within-chunk chunk mass = `1.0` | — |
| `linear_baseline_superchunk_concentration_layer_<L>` | baseline: uniform-within-chunk superchunk mass = `1.0` (chunk ⊂ superchunk) | — |
| `linear_baseline_entropy_layer_<L>` | baseline: uniform-within-chunk entropy = `log(16) ≈ 2.77` | — |
| `version` | `benchmark.VERSION` | — |

**Headline summary**: `hierarchical_robustness_canonical` — the ratio of within-chunk *spread* in late layers (6–11) to within-chunk spread in early layers (0–5), where a layer's spread is `chunk_concentration / local_concentration` (mass over the whole 16-token chunk relative to the tight ±2 local window). A value > 1 indicates the model shifts from fine (local) pooling in early layers to coarse (chunk-level) pooling in late layers. The spread-to-spread normalization cancels the region-size offset, so depth-invariant attention (e.g. uniform) scores ≈ 1.0 — the raw `chunk(late)/local(early)` ratio would instead award a trivial uniform strawman ≈ 3.5, which is why it is *not* used.

**Per-slice values**: One metric per layer for each concentration type and entropy (12 × 4 = 48 metrics), using the naming pattern above.

**Reference baselines**: `linear_baseline_*` — four families (`local`, `chunk`, `superchunk`, `entropy`), one value per layer, derived analytically from a uniform-within-chunk attention pattern (the "no-mechanism" strawman). Each query spreads mass uniformly over the 16 keys in its own chunk and nothing elsewhere; the values are layer-agnostic constants (`0.289`, `1.0`, `1.0`, `log 16`) repeated under identical layer/head indexing so they overlay directly on the per-layer series.

## Bump Procedure

- `VERSION` in `benchmark.py` increments on any payload key change, metric formula change, or canonical condition change (seq_len, chunk_size, num_layers, num_heads).
- After bumping, update this README's "Payload Contract" and "Metrics" tables in the same commit.