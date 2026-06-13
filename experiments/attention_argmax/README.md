# attention_argmax

## Question

Does the attention head implement an **argmax** over the key–query similarities? That is, given a query vector and a set of key vectors, does the attention distribution place nearly all its mass on the single position with the highest similarity (the "winner"), rather than spreading mass across multiple positions?

This is a foundational mechanistic-interpretability question: many circuit hypotheses (induction heads, copying heads, pointer chains) assume the head behaves as a discrete selector. `attention_argmax` quantifies how close a real head is to that ideal.

---

## Setup

**Synthetic generator only** — no trained model required. The goal constructs a controlled "needle-in-haystack" scenario:

- A single query vector `q ∈ ℝ^d`.
- `N` key vectors `K = [k_1, ..., k_N] ∈ ℝ^{N×d}`.
- `N` value vectors `V = [v_1, ..., v_N] ∈ ℝ^{N×d}`.
- The **ground-truth winner** is `i* = argmax_i (q · k_i)`.

The generator produces batches where the similarity gap between the winner and the runner-up is controllable (the *separation* parameter). This lets us measure the head's selection sharpness as a function of how easy the discrimination task is.

The attention head under test is a pure function:

```python
model_fn(q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray
# q: (d,), K: (N, d), V: (N, d)  →  returns attn_weights: (N,)
```

The head must return a valid probability distribution over the `N` positions (non-negative, sums to 1). The attempt's `main.py` wraps its actual model (or mathematical implementation) to match this signature.

---

## Canonical measurement condition

| Parameter          | Value | Note |
|--------------------|-------|------|
| `d` (head dim)     | 64    | Fixed for all seeds |
| `N` (seq len)      | 32    | Fixed |
| Separation sweep   | `[0.0, 0.5, 1.0, 2.0, 4.0]` | Multiples of the noise std; see `task.py` |
| Noise distribution | `N(0, I_d)` | Keys = signal + noise |
| Winner signal      | `||q|| = 1`, `q · k_{i*} = separation` | Runner-up fixed at 0 |
| Seeds per slice    | 100   | `generate(seed)` called with `seed = base + slice_idx * 100 + rep` |

The **canonical headline metric** is reported at `separation = 2.0` (moderately easy discrimination).

---

## Payload contract

`task.evaluate(model_fn)` returns a `dict` with exactly these keys:

```python
{
    "version": 1,                                      # matches benchmark.VERSION
    "config": {
        "d": 64,
        "N": 32,
        "separations": [0.0, 0.5, 1.0, 2.0, 4.0],
        "seeds_per_slice": 100,
        "canonical_separation": 2.0,
    },
    "sweep": [
        {
            "separation": 0.0,
            "winner_mass_mean": 0.03125,      # mean over 100 seeds
            "winner_mass_std": 0.0012,
            "winner_rank_mean": 16.5,         # 1 = best
            "winner_rank_std": 0.2,
            "entropy_mean": 3.4657,
            "entropy_std": 0.001,
        },
        ... (one dict per separation value)
    ],
    "baselines": {
        "uniform_winner_mass": 0.03125,       # 1/N
        "uniform_entropy": 3.4657,            # log(N)
    }
}
```

All floats are Python `float` (not `np.float32`). The `sweep` list order matches `config["separations"]`.

---

## Metrics

`benchmark.score(payload)` returns a flat `dict[str, float | int]`:

| Metric | Formula | Direction | Interpretation |
|--------|---------|-----------|----------------|
| `version` | `payload["version"]` | — | Contract version |
| `argmax_fidelity_canonical` | `winner_mass_mean` at `separation = 2.0` | **Bigger is better** | Headline summary: how much mass the head puts on the true winner when discrimination is moderately easy. Range `[0, 1]`. |
| `argmax_fidelity_sep_<val>` | `winner_mass_mean` at each separation | **Bigger is better** | Per-slice fidelity. Floats in key use `0p5` format (`sep_0p0`, `sep_0p5`, `sep_1p0`, `sep_2p0`, `sep_4p0`). |
| `argmax_rank_canonical` | `winner_rank_mean` at `separation = 2.0` | **Smaller is better** | Average rank of true winner (1 = perfect). |
| `argmax_rank_sep_<val>` | `winner_rank_mean` at each separation | **Smaller is better** | Per-slice rank. |
| `entropy_canonical` | `entropy_mean` at `separation = 2.0` | **Smaller is better** | Attention entropy (nats). Lower = sharper. |
| `entropy_sep_<val>` | `entropy_mean` at each separation | **Smaller is better** | Per-slice entropy. |
| `selection_robustness` | `argmax_fidelity_sep_0p5 / argmax_fidelity_sep_4p0` | **Bigger is better** | Fidelity at the hard separation relative to the easy one. Measures how gracefully the head degrades: `1` = no degradation, `→0` = collapses under hard discrimination. Range `[0, 1]` for real heads; `NaN` if the easy-separation denominator is 0. |
| `uniform_baseline_fidelity` | `1 / N` | — | Baseline: uniform attention. |
| `lift_over_uniform_canonical` | `argmax_fidelity_canonical - uniform_baseline_fidelity` | **Bigger is better** | Absolute improvement over uniform. |

**Edge cases handled in `benchmark.score`:**
- Empty / non-list `sweep` → `ValueError` (a non-empty sweep is required).
- Zero (or missing) denominator in `selection_robustness` → `NaN`.
- Missing keys → `KeyError` with descriptive message.
- Non-finite values in any sweep field → `ValueError`.

---

## Bump procedure

Bump `benchmark.VERSION` (and this README's payload version) when:

- The `sweep` record schema changes (keys added/removed/renamed).
- The canonical separation changes.
- The metric formulas change (e.g., `selection_robustness` uses a different ratio).
- The `model_fn` signature changes.

Do **not** bump when:
- Adding a new metric that doesn't alter existing ones.
- Adding an optional field to `config` with a default.
- Extending the separation sweep (the `sweep` list is already extensible).

Old `benchmark.json` files remain on disk; the dashboard filters to the highest `version` automatically.