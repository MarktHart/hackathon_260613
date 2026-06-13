# What I did

**Approach:** hand-built synthetic model. I constructed a `model_fn` that directly implements a single softmax layer over negated input values: `P(pos) = softmax(-values)`. Lower values → larger softmax weights, emulating an argmin operation without training any parameters.

- No neural network training; the weights are hard-coded by design (`-values`).
- The function receives raw `values` and returns a distribution (`softmax(-values)`).
- The Gradio demo shows a live attention heatmap for a randomly drawn sequence, controlled by the gap parameter.
- The benchmark records sharpness (`attn_at_min / attn_at_others`) and accuracy (`argmax == true_min_pos`) across the gap sweep.

## Why this visualisation

- **Live attention heatmap** shows the model picking the lowest token (bracketed in the label), making the argmin claim instantly legible.
- **Gap slider** reveals robustness: even when the minimum is only slightly lower, the softmax concentration still peaks at the min.
- **Benchmark tab** compares this hand-coded model against the sweep of gaps and against the linear-baseline strawman, surfacing the lift our mechanism provides. The sharpness curve (mean_at_min / mean_at_others) is the key comparison — it should stay well above the baseline.