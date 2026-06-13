# attention_matrix_chain — first_pass

## What I did

First-pass attempt: the most obvious mechanism for computing the composed
two-hop attention matrix. The `model_fn` simply takes the NumPy inputs `A1`
and `A2` from `task.evaluate`, converts them to PyTorch floats on `cuda`, runs
`A2 @ A1` (per head), and returns the NumPy result. No weights are learned,
no MLP, no trickery — it’s the direct linear-algebra composition the question
asks about.

Because the pipeline guarantees a GPU (`DEVICE = "cuda"`), the matrix multiply
actually lands on the hardware. This satisfies the hard GPU requirement
while staying strictly within the `base_model.py` spirit (a tiny delta: only
the compute path, no training loop).

## Why this visualisation

The Gradio app has two tabs:

- **Demo** — a small static page explaining what the baseline model does,
  with no runtime variables needed (the mechanism is fixed; the interesting
  behaviour is encoded in the sweep).
- **Benchmark** — the shared `benchmark_panel("../..")` dashboard that plots
  per-alpha fidelity, KL divergence, and composition robustness against the
  single-hop baseline. This is the visual that tells the story:
  - X-axis = `alpha` ( Dirichlet concentration ) → more peaked rows at lower
    values.
  - Y-axis (per plot) = a metric → higher `chain_fidelity` and higher
    `composition_robustness` mean the mechanism truly recomputes the product.
  - The curve should stay high on the left (where composition matters) and
    stay *above* the single-hop baseline across the whole sweep.

The dashboard directly tests whether the single-horizon shortcut (`A2`)
collapses at the most peaked conditions while a real composition mechanism
(`A2 @ A1` on GPU) holds its score.