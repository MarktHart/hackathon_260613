# attention_modular_add

## Question

When a transformer head is trained on **modular addition** `(a + b) mod p`, the
known mechanism (Nanda et al., 2023; Zhong et al., 2023) represents the inputs
in a **Fourier basis**: the head's query/key projections concentrate on a small
number of frequencies `k`, and the query phase for `a` is the conjugate of the
key phase for `b`, so that `q·k` peaks on `a + b ≡ const (mod p)`.

This goal measures, for **a single attention head**, *how cleanly its Q/K
projections implement that Fourier mechanism*. The headline metric
`fourier_alignment_canonical` is the mean alignment between the head's
query-side and key-side frequency subspaces across all frequencies.

## Setup

**Real-model probe over a fixed synthetic input grid** — no training happens in
this goal. The attempt provides a *trained* single head as a `model_fn`; the
task feeds it every `(a, b)` pair and analyses the returned Q/K vectors.

- **Task**: modular addition modulo prime `p = 97`.
- **Tokens**: sequence length 3, `[a, b, =]`, where `a, b ∈ [0, p-1]` and the
  `=` separator is the token whose id equals `p` (i.e. `97`).
- **Evaluation set**: the full Cartesian product of all `p² = 9409` pairs
  `(a, b)`, generated deterministically (see `generate`).
- **What is analysed**: the head's **query** vector at the `a` position and its
  **key** vector at the `b` position, swept against every Fourier frequency
  `k = 1 .. p//2 = 48`.

## Canonical Measurement Condition

| Parameter | Value | Source |
|-----------|-------|--------|
| Prime modulus `p` (`modulus`) | 97 | `task.P` |
| Head dimension `d_head` | 128 | `task.D_HEAD` |
| Sequence length | 3 (`a`, `b`, `=`) | fixed |
| `=` token id | 97 (`= p`) | fixed |
| Evaluation samples | 9409 (`= p²`, all pairs) | `generate` |
| Frequencies swept `k` | 1 .. 48 (`= p//2`) | `_compute_sweep` |
| Head analysed | `layer_index = 0`, `head_index = 0` | fixed |

`generate(seed)` accepts a `seed` but **ignores it** — the canonical condition
is fully fixed (the entire `(a, b)` grid), so any seed yields the identical
`Batch`. This is deterministic by construction.

## Model Function Signature

Every attempt must provide a `model_fn` with this exact signature:

```python
def model_fn(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        tokens: int array of shape (batch, 3). Columns are [a, b, 97].
                Column 0 = a, column 1 = b, column 2 = p (the '=' token).

    Returns:
        (Q, K), where:
            Q: float array of shape (batch, 3, d_head). Per-position query
               vectors for the head under test (d_head = 128).
            K: float array of shape (batch, 3, d_head). Per-position key
               vectors for the same head.
    """
```

**Constraints**:
- Pure NumPy interface. The attempt's `main.py` wraps its PyTorch/JAX head.
- No gradients, no training, no I/O inside `model_fn`.
- `batch == 9409` for the canonical run (the full grid is passed in one call).
- `Q`/`K` must have shape exactly `(9409, 3, 128)`; `evaluate` raises
  `ValueError` otherwise.
- Deterministic for fixed weights (no dropout at eval time).

`task.evaluate` only uses `Q[:, 0, :]` (query at the `a` position) and
`K[:, 1, :]` (key at the `b` position); the `=` position is included in the
contract for shape uniformity but is not analysed.

## Payload Contract

`task.evaluate(model_fn)` returns a `dict` with the following keys. The keys
`version, modulus, layer_index, head_index, d_head, sweep` are **required** by
`benchmark.score` (a missing one raises `KeyError`). The remaining keys are
self-describing metadata that `score` recomputes from `sweep` rather than
trusting directly.

```python
{
    # ---- Required by benchmark.score ----
    "version": 1,                 # int, payload schema version (== benchmark.VERSION)
    "modulus": 97,                # int, prime p
    "d_head": 128,                # int, head dimension (used for the baseline)
    "layer_index": 0,             # int, which layer the analysed head is in
    "head_index": 0,              # int, which head within that layer
    "sweep": [                    # list of p//2 == 48 records, one per frequency
        {
            "frequency": 1,              # int k, 1 .. 48
            "alignment": 0.93,           # float in [0, 1], mean cosine of principal
                                         #   angles between the Q-side and K-side
                                         #   freq-k subspaces (1 = perfectly aligned)
            "phase_error": 0.07,         # float in [0, pi], magnitude-weighted
                                         #   deviation from the conjugate-phase
                                         #   prediction w_K = conj(w_Q) (smaller better)
            "explained_variance": 0.41,  # float in [0, 1], fraction of centered
                                         #   ||Q||^2 + ||K||^2 captured by freq k
        },
        ...
    ],

    # ---- Self-describing metadata (recomputed by score, not trusted) ----
    "total_explained_variance": 0.88,   # float, sum of explained_variance over sweep
    "max_alignment": 0.93,              # float, max alignment across frequencies
    "argmax_alignment_freq": 1,         # int, frequency with the max alignment
}
```

### How each per-frequency quantity is computed (inside `evaluate`)

For frequency `k`, the centered query vectors `Q_c` are regressed onto the
`(sin_k, cos_k)` features of `a`, giving a `2 × d_head` matrix `W_Q`; likewise
`W_K` for the key vectors against the features of `b`.

- **`alignment`** — singular values (cosines of principal angles) of
  `Vt_Q @ Vt_Kᵀ`, where `Vt_Q`/`Vt_K` are the right singular vectors of
  `W_Q`/`W_K`. Mean over the (≤2) angles, clipped to `[0, 1]`.
- **`phase_error`** — treat each `W` row pair as a complex direction
  `w = W[0] + i·W[1]`; the magnitude-weighted mean of `|angle(w_Q · conj(w_K))|`,
  clipped to `[0, π]`. Zero means the conjugate-phase relation holds exactly.
- **`explained_variance`** — energy of the least-squares freq-`k` reconstruction
  of `Q_c` and `K_c` (normalised by column energy so it is a true *fraction* of
  centered variance), divided by `||Q_c||² + ||K_c||²`, clipped to `[0, 1]`.

## Metrics

`benchmark.score(payload)` returns a flat `dict[str, float | int]`.

| Metric | Formula / Meaning | Range | Direction | Role |
|--------|-------------------|-------|-----------|------|
| `version` | `payload["version"]` (`== VERSION`) | int | — | Dashboard filter |
| `fourier_alignment_canonical` | `mean(alignment)` over all frequencies | [0, 1] | bigger | **Headline** |
| `phase_error_canonical` | `mean(phase_error)` over all frequencies | [0, π] | **smaller** | Mechanism quality |
| `explained_variance_canonical` | `mean(explained_variance)` over all frequencies | [0, 1] | bigger | Coverage |
| `random_baseline_alignment` | `2 / d_head` (analytic chance alignment) | [0, 1] | — | Baseline |
| `lift_over_random_alignment` | `fourier_alignment_canonical − random_baseline_alignment` | [-1, 1] | bigger | Headline vs baseline |
| `superposition_robustness` | `min(alignment) / max(alignment)` (`0` if `max == 0`) | [0, 1] | bigger | Across-frequency ratio |
| `fourier_alignment_freq_{k:02d}` | `alignment` for frequency `k` | [0, 1] | bigger | Per-slice |
| `phase_error_freq_{k:02d}` | `phase_error` for frequency `k` | [0, π] | **smaller** | Per-slice |

**Direction of better**: everything is bigger-is-better **except**
`phase_error_canonical` and the per-slice `phase_error_freq_*`, which are
smaller-is-better. The dashboard does not encode direction; the grader infers
it from the name.

Per-slice keys use a zero-padded integer frequency (`fourier_alignment_freq_01`
… `fourier_alignment_freq_48`). The axis here is an integer frequency, so the
`0p7`-style float encoding from the framework guide does not apply.

## Baselines

- **`random_baseline_alignment` = `2 / d_head` ≈ 0.0156** at `d_head = 128`.
  This is the analytic chance alignment for two random 2-D subspaces in
  `ℝ^d_head`; it is deliberately conservative (Gaussian-noise heads score above
  it empirically), so a head that fails to beat it carries essentially no
  Fourier structure. Computed in `benchmark.py` — no extra model run needed.
- **`random_model_fn()`** in `task.py` returns contract-shaped Gaussian-noise
  `Q, K` (seeded off the token sum, hence reproducible). It exists for smoke
  testing the wiring, not as a scored baseline.

## `is_obviously_broken`

`benchmark.is_obviously_broken(metrics)` short-circuits the jury (marks the
attempt failed) only for clearly degenerate results:

- any `NaN`/`inf` metric, or
- `fourier_alignment_canonical <= random_baseline_alignment`.

It deliberately does **not** threshold on mean `explained_variance` or
`phase_error`: a clean few-frequency head concentrates its variance on a couple
of frequencies, so those means are legitimately small.

## Pipeline hooks

- `GPU_REQUIREMENT = 1` — every attempt runs on the GPU; the head is evaluated
  there even though `task.py`/`benchmark.py` themselves are pure CPU/NumPy.

## Bumping `VERSION`

`VERSION = 1` lives in `benchmark.py`. **Bump it (and update this README in the
same commit) when:**
- any metric formula changes (e.g. the baseline `2/d_head`, the
  `superposition_robustness` ratio, or an alignment/phase definition);
- a required payload key is added, removed, or retyped;
- the canonical condition changes (`p`, `d_head`, swept frequencies, which head
  is analysed).

**Do NOT bump when:**
- adding a new per-slice metric (e.g. `explained_variance_freq_{k}`);
- adding an optional payload key with a default in `score()`;
- fixing a bug in `evaluate` that does not change the contract.

Old `benchmark.json` files stay on disk; the dashboard filters to the highest
`version` present.

## Anti-Patterns This Goal Avoids

- **Raw tensors in payload**: `evaluate` reduces the `9409 × 3 × 128` Q/K
  activations to 48 small per-frequency records.
- **Per-attempt branching**: `score()` only reads `payload`; it never inspects
  the attempt name or files.
- **Silent contract changes**: required keys are validated and `VERSION` gates
  the dashboard.
