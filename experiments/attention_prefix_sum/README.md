# Attention Prefix Sum

## Question
Can attention mechanisms natively compute prefix sums (cumulative sums) over a sequence without explicit recurrence or MLP assistance? This tests whether the attention matrix can implement the triangular masking pattern required for prefix accumulation — a canonical algorithmic primitive.

## Setup
**Synthetic generator only.** No trained models. The task generates sequences of integer tokens and evaluates whether a model's forward pass produces the correct prefix sum at each position.

- Vocabulary: integers `0..V-1` (default `V=10`)
- Sequence length `L` varied in sweep: `[4, 8, 16, 32, 64]`
- Target at position `t`: `(sum_{i=0..t} x_i) mod V`
- Training signal: cross-entropy on next-token prediction (prefix sum at each position)
- Canonical measurement condition: `L=16`, `V=10`, 512 sequences, greedy decoding

## Canonical measurement condition
Every attempt must evaluate on:
- `seq_len = 16`
- `vocab_size = 10`
- `num_sequences = 512`
- `seed = 0` (deterministic batch)
- Metric reported at this condition is the **headline** `prefix_acc_canonical`

## Model function signature
```python
def model_fn(input_ids: np.ndarray) -> np.ndarray:
    """
    Args:
        input_ids: int32 array of shape [batch, seq_len], values in 0..vocab_size-1
    Returns:
        logits: float32 array of shape [batch, seq_len, vocab_size]
                Unnormalised logits for the prefix-sum target at each position.
    """
```
The attempt's `main.py` must provide a callable matching this signature. `task.evaluate` calls it exactly once per batch.

## Payload contract
`task.evaluate` returns a `dict` with the following keys — **this is the exact shape `benchmark.score` consumes**:

```python
{
    "version": 1,                                    # int, mirrors benchmark.VERSION
    "config": {                                      # self-describing, not used by score()
        "seq_len": 16,
        "vocab_size": 10,
        "num_sequences": 512,
        "seed": 0,
    },
    "sweep": [                                       # one record per seq_len in [4,8,16,32,64]
        {
            "seq_len": 4,
            "correct": 2048,                         # int, total correct token predictions
            "total": 2048,                           # int, total token positions evaluated
        },
        {
            "seq_len": 8,
            "correct": 4096,
            "total": 4096,
        },
        {
            "seq_len": 16,
            "correct": 8192,
            "total": 8192,
        },
        {
            "seq_len": 32,
            "correct": 16384,
            "total": 16384,
        },
        {
            "seq_len": 64,
            "correct": 32768,
            "total": 32768,
        },
    ],
    "random_baseline_correct": 819,                  # int, expected correct by random guessing at canonical condition (8192 // 10)
    "random_baseline_total": 8192,                   # int, total positions at canonical condition
}
```
All `correct`/`total` are **aggregated over the full batch** (not per-sequence averages). `random_baseline_*` are computed analytically: `total / vocab_size`.

## Metrics
`benchmark.score` returns a flat `dict[str, float | int]`:

| metric | formula | bigger is better? |
|--------|---------|-------------------|
| `version` | `benchmark.VERSION` | — |
| `prefix_acc_canonical` | `sweep[seq_len=16].correct / sweep[seq_len=16].total` | ✓ |
| `prefix_acc_len_4` | `sweep[seq_len=4].correct / sweep[seq_len=4].total` | ✓ |
| `prefix_acc_len_8` | `sweep[seq_len=8].correct / sweep[seq_len=8].total` | ✓ |
| `prefix_acc_len_16` | alias of `prefix_acc_canonical` | ✓ |
| `prefix_acc_len_32` | `sweep[seq_len=32].correct / sweep[seq_len=32].total` | ✓ |
| `prefix_acc_len_64` | `sweep[seq_len=64].correct / sweep[seq_len=64].total` | ✓ |
| `prefix_acc_robustness` | `min(prefix_acc_len_*) / prefix_acc_canonical` ∈ [0,1] | ✓ |
| `linear_baseline_acc_canonical` | `random_baseline_correct / random_baseline_total` (= 1/vocab_size) | — |
| `lift_over_baseline_canonical` | `prefix_acc_canonical - linear_baseline_acc_canonical` | ✓ |

Direction: **accuracy metrics are bigger-is-better**. `prefix_acc_robustness` measures whether performance degrades at longer lengths; 1.0 = no degradation.

## Bump procedure
- `VERSION` in `benchmark.py` **must** be incremented if:
  - any metric formula changes;
  - payload keys are added/removed/retyped;
  - canonical `seq_len` or `vocab_size` changes.
- Update this README's payload contract and metrics table in the same commit.
- Old `benchmark.json` files remain on disk; dashboard filters to highest `version`.