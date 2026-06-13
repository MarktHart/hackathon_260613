# attention_group_compose

## Question
Do attention matrices in transformers compose according to group-theoretic laws? Specifically, when attention heads implement permutations (e.g., induction heads copying tokens), does their composition follow the algebraic structure of the symmetric group — closure, associativity, identity, and inverses — or does the softmax relaxation break these symmetries in measurable ways?

## Setup
Synthetic generator. We construct attention matrices as **noisy permutation matrices** representing elements of the cyclic group C_n (rotations). The ground-truth composition is exact matrix multiplication (which corresponds to group addition modulo n). Noise is injected by adding i.i.d. Gaussian noise to logits before softmax, controlled by a temperature parameter.

- Group: Cyclic group C_n (n positions, rotation by k steps)
- Noise model: `attn = softmax((log(P) + σ·ε) / τ)` where P is a permutation matrix, ε ~ N(0,1), τ=1 fixed. The clean logits log(P) are approximated by ±L with L=20, so σ is in **logit units**; σ is scaled relative to L so the sweep spans clean→chance.
- Sweep axis: `noise_level` = σ ∈ {0.0, 10.0, 20.0, 30.0, 40.0}
- Canonical condition: n=6, noise_level=20.0, 200 random pairs per noise level.

At σ=0 the matrices are exact hard permutations (baseline error 0, true sanity check); at σ=20 (canonical) the noisy matrices are meaningfully corrupted so naive matmul is clearly wrong, leaving headroom for a structure-aware method; at σ=40 errors approach chance.

## Canonical measurement condition
Every attempt must evaluate at n=6, noise_level=20.0 with 200 composition queries (pairs of group elements). The sweep across all five noise levels is required for robustness metrics.

## Model function signature
```python
def model_fn(attn_a: np.ndarray, attn_b: np.ndarray) -> np.ndarray:
    """
    Predict the composed attention matrix C ≈ A @ B.
    
    Args:
        attn_a: [n, n] row-stochastic matrix (first group element)
        attn_b: [n, n] row-stochastic matrix (second group element)
    Returns:
        [n, n] row-stochastic matrix: predicted composition.
    """
```
The attempt's `main.py` implements this function using its interpretation method (e.g., SVD projection to permutation matrices, Lie algebra log/exp, direct multiplication, etc.).

## Payload contract
`task.evaluate` returns a dict with exactly these keys:
```python
{
    "version": 1,
    "config": {
        "group": "cyclic",
        "n": 6,
        "noise_levels": [0.0, 10.0, 20.0, 30.0, 40.0],
        "num_pairs_per_level": 200,
        "seed": 0
    },
    "sweep": [
        {
            "noise_level": 0.0,
            "frobenius_error": float,          # ||pred - true||_F / n
            "linear_baseline_error": float,    # ||A @ B - true||_F / n (naive matmul)
            "num_pairs": 200
        },
        ... (one per noise_level)
    ]
}
```
- `frobenius_error`: Mean Frobenius norm of (predicted - true_composition) normalized by n.
- `linear_baseline_error`: Same metric for the naive baseline `pred = attn_a @ attn_b` (matrix multiply the noisy matrices directly).
- All errors are ≥ 0. Lower is better.

## Metrics
`benchmark.score(payload)` returns a flat dict:

| Metric | Formula | Direction | Description |
|--------|---------|-----------|-------------|
| `version` | payload["version"] | — | Benchmark version |
| `composition_fidelity_canonical` | 1 - frobenius_error_at_noise_20p0 | Bigger ✓ | Headline: accuracy at canonical noise (σ=20) |
| `composition_fidelity_noise_0p0` | 1 - frobenius_error_at_noise_0p0 | Bigger ✓ | Perfect permutations (sanity) |
| `composition_fidelity_noise_10p0` | 1 - frobenius_error_at_noise_10p0 | Bigger ✓ | Low noise |
| `composition_fidelity_noise_20p0` | 1 - frobenius_error_at_noise_20p0 | Bigger ✓ | Canonical (duplicate for dropdown) |
| `composition_fidelity_noise_30p0` | 1 - frobenius_error_at_noise_30p0 | Bigger ✓ | Medium noise |
| `composition_fidelity_noise_40p0` | 1 - frobenius_error_at_noise_40p0 | Bigger ✓ | High noise (approaches chance) |
| `linear_baseline_fidelity_noise_*` | 1 - linear_baseline_error_at_* | Bigger ✓ | Naive matmul baseline per slice (`0p0`,`10p0`,`20p0`,`30p0`,`40p0`) |
| `lift_over_baseline_canonical` | fidelity_canonical - linear_baseline_fidelity_noise_20p0 | Bigger ✓ | Improvement over naive matmul |
| `composition_robustness` | (AUC of fidelity curve over noise, trapezoid rule) / noise_span | Bigger ✓ | Span-normalized area under fidelity curve, in [0,1] |

All fidelity metrics are in [0, 1] (clipped). `composition_robustness` = 1 means perfect at all noise levels; 0 means chance-level.

## Bump procedure
Bump `VERSION` in `benchmark.py` if:
- The error normalization changes (e.g., divide by n vs n²)
- A noise level is added/removed from the canonical sweep
- The payload keys are renamed or retyped
- The group changes (e.g., C_n → S_n)

Adding a new metric without changing existing ones does **not** require a version bump.