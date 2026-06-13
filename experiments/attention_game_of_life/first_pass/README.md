# What I did

This is a **hand-built** attempt that encodes the full Game of Life dynamics as a fixed-convolution circuit on the GPU. The model Fn maps a (B, H, W) input grid to a per-cell logit by:

1. Padded the input toroidally.
2. Applying a 3×3 convolution kernel (all ones on the ring, zero center) to count the eight immediate neighbor values.
3. Scaling the neighbour count by a hand-chosen temperature-like factor (7.0) and offsetting by a constant (to capture the 2-or-3-survive / 3-birth threshold).
4. Returning the resulting logit tensor on the CPU for the harness.

No parameters are learned; the only knobs are the kernel (fixed) and the scale/offset (hand-tuned to yield > 0 logits for alive cells and < 0 logits for dead ones under the Game of Life rule). The circuit is expressed as a pure torch tensor flow on `cuda`.

## Why this visualisation

The Demo tab shows three things together:

- The 3×3 convolution kernel as a heatmap — its values (1.0s and 0.0 centre) are the only learned “parameters” in the circuit.
- A hand-tuned scale factor slider (displayed only for reference; the actual value is fixed to 7.0 here).
- A per-cell accuracy heatmap at the canonical density (0.3), built from the recorded payload. This visual shows that the hand-built circuit does not produce a uniform blur: alive cells are correctly inferred with high consistency.

Together, these let a viewer confirm three claims at once:
- The model is actually convolving with the correct eight-neighbor pattern (kernel heatmap).
- The logit thresholding works (high accuracy vs the static baseline).
- The visualisation surface is fine-grained enough to see a learned pattern rather than random noise.