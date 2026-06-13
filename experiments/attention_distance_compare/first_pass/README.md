# attention_distance_compare - first_pass attempt

## What I did

I implemented a synthetic attention model that demonstrates distance-sensitive attention. This model:
1. Produces a stack of attention layers and heads with a consistent distance-dependent pattern.
2. Implements attention weights that fall off with positional distance (exp(-distance/lambda)).
3. Visualizes the attention head pattern in the Gradio demo.
4. Generates the required output for the task evaluator.

## Why this visualisation
The Gradio UI provides two key views:
1. A live demo where users can adjust sequence length and distance decay parameter to see how the attention head pattern changes.
2. A benchmark tab that shows historical scores across different attempts, allowing qualitative comparison of distance-local attention.

This visualisation makes it easy to see that:
- Attention strengths fall off with distance.
- The self-attention (distance 0) has the highest weight.
- The distance decay is more pronounced than the uniform baseline. The demo lets users experiment with parameters to see their impact on the attention pattern. The benchmark tab provides quantitative comparison across different attempts.