# attention_xor — first_pass

### What I did
I implemented a pure NumPy `model_fn` that extracts the two binary features A and B from the token matrix, then computes the hand-built XOR logit:

    logit = 2*(A + B - 2*AB) - 1 = (A ^ B ? 1 : 0) - 0.5   scaled as 2*(XOR - 0.5)

This satisfies the `task.evaluate` contract (input `(N,4) int32` tokens, output `(N,) float32` logits). I also baked a demo version of the same function inside a tiny `torch.nn` wrapper so the Dashboard can compare hand-built logic against a "learned" neural equivalent in the same UI.

No training is performed — the weights are literal constants (2, 2, -4, -1) chosen to map A XOR B to +1 when True and -1 when False. The Gradio demo visualises per-token breakdown of A, B, AB, and the final logit.

### Why this visualisation
The two-panel demo tells a clear story:
- The token table shows the ground-truth feature values for each batch token.
- The logit histogram shows the distribution of predicted scores, with a red line at 0 marking the decision threshold.
- The tiny-MLP toggle demonstrates that the same functional behaviour can live inside a neural block, while the hand-built version proves the weights are manually intelligible.

The Benchmark tab overlays the same metric (lift over linear baseline) across all runs under the `attention_xor` goal, so future attempts can measure how much more robust or noise-resilient they are relative to this pure NumPy baseline.

Metrics for this run will be dominated by the gap between hand-built XOR (near-perfect) and the optimal linear probe (≈75% at p=0.5), confirming the model is implementing the non-linear XOR operation — not a linear approximation.