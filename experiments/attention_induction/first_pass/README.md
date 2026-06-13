What I did
- Adapted a HuggingFace causal language model (gpt2-medium) to produce the required `ModelFn` signature.
- Extracted only the canonical induction head's attention (layer=11, head=7) and returned both attention and logits as NumPy arrays.
- Wrapped the HF model output in a function that mirrors the `task.evaluate` contract, then ran the canonical sweep to obtain the scoring payload.
- Saved a CSV of the sweep per-head records and wrote the benchmark JSON to `results/`.

Why this visualisation
- The demo tab shows the headline metric (induction_selectivity) together with the exact induction score of the canonical head, giving a single‑sentence statement of performance.
- Using a dropdown to select any CSV under `results/` lets a human observer toggle between runs, compare per‑head induction scores, and verify that the model actually places mass on the induction source position.
- The Benchmark tab pulls in the goal’s shared leaderboard, showing the metric across all attempts at once.