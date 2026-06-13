# attention_kth_select

## Question

Can an attention head implement a **k-th position selection** operation — attending sharply to the token at a specific position *k* in the sequence, regardless of token content?

This is a foundational mechanistic capability: position-based addressing (e.g., "attend to the 3rd token") underlies induction, copying, and many algorithmic tasks. We test whether a head's attention pattern peaks at the designated position *k* across varied sequence lengths and distractor tokens.

---

## Setup

**Synthetic generator only.** No trained model required.

- Sequences are random token IDs drawn uniformly from the **full** vocabulary of size `V = 100` (IDs `0..99`).
- Each sequence has a **marker token** `M = 99` forced at position `k` (0-indexed); every other position is random noise over the same full vocabulary, so the marker value `99` *also* occurs at other positions by chance.
- The task: attend to position `k`. Because the marker value is not reserved — it appears elsewhere too — position `k` is **not** uniquely identifiable by content. A content-matching head (attend to value `99`) cannot reliably isolate `k`; only positional information addresses it cleanly.
- Sequence length `L = 32`. We sweep `k ∈ {0, 4, 8, 12, 16, 20, 24, 28}` (8 positions, spaced by 4).
- Batch size `B = 128` sequences per `k`.
- Canonical measurement condition: `k = 8` (1/4 into the sequence, well clear of boundaries).

The generator is deterministic given `seed`. Same seed → same batch.

---

## Canonical Measurement Condition

| Parameter | Value |
|-----------|-------|
| Sequence length `L` | 32 |
| Vocabulary size `V` | 100 |
| Marker token `M` | 99 |
| Sweep positions `k` | [0, 4, 8, 12, 16, 20, 24, 28] |
| Canonical `k` | 8 |
| Batch size per `k` | 128 |
| Total sequences | 1024 |

Every attempt **must** evaluate on this exact sweep. The `task.generate` function accepts `seed` but the canonical sweep is fixed; the seed only affects the random noise tokens.

---

## Model Function Signature

```python
def model_fn(
    input_ids: np.ndarray,      # shape (B, L), int32, token IDs
    positions: np.ndarray,      # shape (L,), int32, 0..L-1
) -> np.ndarray:                # shape (B, L), float32, attention weights over positions for a *single* head
    ...
```

- `input_ids[b, l]` is the token at batch `b`, position `l`.
- `positions[l] = l` (provided for convenience; some attempts may use relative positions).
- Return value: **attention weights** for one head, shape `(B, L)`, non-negative, summing to 1 over `L` for each batch item.
- The attempt chooses *which* head to expose (e.g., the head it claims does k-th selection). `evaluate` calls `model_fn` once per `k`-condition with the corresponding batch.

---

## Payload Contract

`task.evaluate` returns a dict with exactly these keys:

```python
{
    "version": 1,                           # payload schema version
    "canonical_k": 8,                       # the canonical sweep value
    "sweep": [                              # length 8, one per k in [0,4,8,12,16,20,24,28]
        {
            "k": int,                       # the target position
            "attn_at_k": float,             # mean attention weight on position k across the batch
            "attn_entropy": float,          # mean entropy of attention distribution (nats)
            "attn_max_pos": float,          # mean position of max attention weight
            "batch_size": int,              # 128
        },
        ...
    ],
    "model_name": str,                      # filled by attempt (e.g. "gpt2-small-layer5-head3")
    "dataset": "synthetic_kth_select",      # fixed identifier
}
```

All floats are Python `float` (not numpy scalars). `attn_at_k` ∈ [0, 1]. `attn_entropy` ≥ 0. `attn_max_pos` ∈ [0, L-1].

---

## Metrics

`benchmark.score` returns a flat dict:

| Metric | Formula | Direction | Meaning |
|--------|---------|-----------|---------|
| `kth_select_accuracy_canonical` | `attn_at_k` at `k=8` | bigger = better | Headline: how much mass the head puts on the correct position at the canonical condition. |
| `kth_select_accuracy_k_<k>` | `attn_at_k` for each `k` in sweep | bigger = better | Per-slice: does the head work at all positions? |
| `kth_select_sharpness_canonical` | `1 - attn_entropy / log(L)` at `k=8` | bigger = better | Normalised concentration: 1 = delta spike, 0 = uniform. |
| `kth_select_sharpness_k_<k>` | same for each `k` | bigger = better | Per-slice sharpness. |
| `kth_select_position_bias` | `mean(abs(attn_max_pos - k))` over sweep | smaller = better | Average positional error (in tokens). |
| `linear_baseline_accuracy_canonical` | `1 / L` ≈ 0.03125 | — | Uniform attention baseline at canonical `k`. |
| `lift_over_linear_baseline_canonical` | `kth_select_accuracy_canonical - linear_baseline_accuracy_canonical` | bigger = better | Improvement over uniform. |

`version` is always the first key in the returned dict (matches `benchmark.VERSION`).

---

## Bump Procedure

- `VERSION` in `benchmark.py` and `version` in payload **must** be bumped together when:
  - Any metric formula changes.
  - A payload key is added, removed, or retyped.
  - The canonical `k` or sweep values change.
  - The `model_fn` signature changes.
- Adding a new metric without touching existing ones does **not** require a bump.
- After bumping, update this README's "Payload Contract" and "Metrics" tables in the same commit.

---

## Smoke Test

The pipeline runs:

```python
from experiments.attention_kth_select import task, benchmark
payload = task.evaluate(task.random_model_fn())
metrics = benchmark.score(payload)
assert "kth_select_accuracy_canonical" in metrics
```

`random_model_fn` returns uniform attention (`1/L` everywhere). Expected metrics:
- `kth_select_accuracy_canonical` ≈ 0.03125
- `kth_select_sharpness_canonical` ≈ 0.0
- `linear_baseline_accuracy_canonical` = 0.03125
- `lift_over_linear_baseline_canonical` ≈ 0.0