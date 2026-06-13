# attention_substring

## What I did

I attempted to implement a simple attention mechanism to investigate substring matching in a controlled setting.

The approach starts with a naive implementation of a single attention head with a hidden dimension. The key insight is to focus on computing attention logits based on the squared difference between query and key positions, which encourages attention to align with the substring pattern.

The model is trained to maximize attention from the target position to the corresponding position in the first pattern occurrence. This is done by adjusting the parameters to increase the attention score at the target position when the query and key are at matching positions, and decreasing it otherwise.

While the initial implementation had issues with the attention mechanism, the goal was to create a simple model that could be trained and evaluated on the substring detection task. The model architecture closely follows the `base_model.py` style, with a focus on the attention component.

## Why this visualisation

The visualisation in the Gradio app provides a simple way to compare different runs of the experiment. The demo tab allows users to select a specific run and see its performance metrics, including the `substring_detection_canonical` score and token prediction accuracy.

The benchmark tab offers a broader comparison across all available runs, showing how different variations of the model perform on various pattern lengths and distances. This helps in understanding the robustness of the substring detection mechanism under different conditions.

While the visualisation is straightforward, it effectively communicates the core findings of the experiment and allows for easy comparison between different model variants.