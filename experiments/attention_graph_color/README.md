# attention_graph_color

## Question

Can an attention mechanism represent the structure of a **proper graph
coloring**? A proper coloring assigns colours to nodes so that no edge
connects two same-coloured nodes. We ask whether a model's attention matrix
respects that constraint: does it place *more* attention on differently
coloured node pairs than on same-coloured ones — and especially on the edges
that (correctly) connect different colours rather than the non-edges?

## Setup

**Synthetic generator only** — no trained model.

We generate Erdős–Rényi graphs G(n, p) and compute a guaranteed-proper greedy
coloring (highest-degree-first, k = number of colours used). Each node gets a
feature vector: a one-hot colour indicator (k dims) plus a normalised degree
scalar, for (k+1) dims total.

- Graph sizes: `n ∈ {20, 40, 60}`
- Edge probabilities: `p ∈ {0.1, 0.2, 0.3}`
- 5 graphs per (n, p) combination → **45 graphs** total
- Coloring: greedy, descending-degree order, proper by construction

### Model function signature

```python
model_fn(adj: np.ndarray, feats: np.ndarray) -> np.ndarray
```

- `adj`: `(n, n)` symmetric adjacency matrix, float32, entries in `{0, 1}`,
  zero diagonal.
- `feats`: `(n, k+1)` node features, float32 (one-hot colour + normalised
  degree).
- Returns: `(n, n)` attention-weight matrix, float32. Rows for non-isolated
  nodes should be non-negative; isolated-node rows may be zeros.
  Normalisation is **not** required — the evaluator reads only relative mass.

## Canonical Measurement Condition

- Canonical graph size: `n = 40` — the only axis the metrics slice on. Every
  headline metric averages over **all** n=40 graphs (all 3 edge probabilities,
  15 graphs / 5 per p).
- `canonical_p = 0.2` is recorded in the payload as nominal metadata only; it
  is **not** used to filter the canonical slice (the sweep records carry no
  `p` field, and `benchmark.score` slices solely by `num_nodes`).
- Evaluation seed: `0` (deterministic generation; `generate(seed)` is
  reproducible for any seed).
- Every graph is evaluated independently; the payload aggregates per graph and
  `benchmark.score` slices by `n`.

## Payload Contract

`task.evaluate(model_fn)` returns a dict with exactly these keys:

```python
{
    "version": 1,
    "model_name": "synthetic_attention_graph_color",
    "canonical_n": 40,
    "canonical_p": 0.2,
    "n_values": [20, 40, 60],
    "num_graphs": 45,
    "sweep": [                                  # one record per graph
        {
            "graph_idx": 0,
            "num_nodes": 20,
            "num_colors": 4,                    # k = colours used
            "edge_density": 0.10,               # |E| / (n choose 2)
            "same_color_attention": 0.012,      # mean attn over same-colour i<j pairs
            "diff_color_attention": 0.013,      # mean attn over diff-colour i<j pairs
            "cross_edge_same_color": 0.0,       # mean attn on same-colour EDGES (exactly 0: proper coloring has none; sanity invariant)
            "cross_edge_diff_color": 0.05,      # mean attn on diff-colour EDGES
            "isolated_node_fraction": 0.0,      # fraction of degree-0 nodes
        },
        ...
    ],
    "baseline_sweep": [                         # no-mechanism (uniform) reference, same graphs
        {
            "graph_idx": 0,
            "num_nodes": 20,
            "same_color_attention": ...,
            "diff_color_attention": ...,
            "cross_edge_same_color": ...,
            "cross_edge_diff_color": ...,
            ...
        },
        ...
    ],
}
```

All attention statistics are means over the relevant set of unordered pairs
(`i < j`). `same_/diff_color_attention` average over *all* pairs of that
colour relation; `cross_edge_*` average over **edges only** (`adj[i,j]=1`).
The `baseline_sweep` records are computed by the task itself from a uniform
attention matrix (`1/(n-1)` off-diagonal) — the structureless strawman —
so the benchmark can report a like-for-like reference.

## Metrics

Returned by `benchmark.score(payload)` — flat dict of scalars.

| Metric | Formula | Direction | Interpretation |
|--------|---------|-----------|----------------|
| `color_separation_canonical` | mean over n=40 graphs of `diff_color_attention − same_color_attention` | bigger better | **Headline.** How much more attention goes to differently- vs same-coloured pairs |
| `color_separation_n_<n>` | same, per graph size n | bigger better | Per-size slice |
| `edge_respect_canonical` | mean over n=40 of `cross_edge_diff_color − cross_edge_same_color` | bigger better | On edges, preference for proper (diff-colour) over improper (same-colour) endpoints |
| `edge_respect_n_<n>` | same, per n | bigger better | Per-size slice |
| `invalid_edge_attention_canonical` | mean over n=40 of `cross_edge_same_color` | smaller better | Attention mass on coloring-violating edges. The greedy coloring is proper by construction, so no same-colour edges exist and this is **identically 0** for every attempt — a sanity invariant, not a discriminating metric |
| `color_separation_overall` | mean separation over all 45 graphs | bigger better | Cross-condition summary |
| `linear_baseline_color_separation` | separation of the uniform-attention baseline at n=40 (≈ 0) | — | No-mechanism reference |
| `lift_over_linear_baseline` | `color_separation_canonical − linear_baseline_color_separation` | bigger better | Improvement over the strawman |
| `num_graphs` | graph count in the sweep | — | Bookkeeping |
| `version` | `VERSION` | — | Dashboard version filter |

The headline summary is `color_separation_canonical`. A uniform attention
matrix gives separation ≈ 0 (same and diff pairs receive equal mass), so any
positive lift indicates the model has learned the colour distinction.

### Edge cases

- Empty `n` slices average to `0.0` (no division by zero).
- A pair/edge class with no members contributes `0.0` for that graph.
- `is_obviously_broken` flags NaN/inf metrics and any attempt that fails to
  beat the uniform baseline at the canonical condition (skips the jury only).

## Bump Procedure

Bump `VERSION` in `benchmark.py` when: a metric formula changes, a payload key
is renamed/retyped/removed, or the canonical condition changes. Update this
README's payload contract and metric table in the same commit. Old
`benchmark.json` files stay on disk; the dashboard filters to the highest
version present.
