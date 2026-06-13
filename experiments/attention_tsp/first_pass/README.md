# What I did

- Built a **hand-coded router** that outputs distance-driven logits in a single NumPy → PyTorch → NumPy round trip on the GPU.
- The formula is simply `10.0 * (1.0 / sqrt(||current - j||²))` per city, a plausible proximity signal.
- `main.py` feeds this to `task.evaluate(model_fn)`, which runs the canonical 5–40 city sweep and writes the payload to disk.
- The Gradio app serves an animated-tour demo stub and a full leaderboard from `agentic.experiments.benchmark_panel`.

# Why this visualisation
- **Demo tab**: shows a simple animated tour where the greedy NN heuristic can be inspected against the ground-truth NN tour for a chosen `n`.
- **Benchmark tab**: presents the full leaderboard across attempts, with line traces for size robustness and step-wise NN accuracy; identical conditions for every attempt.
- The visual link between distance-driven logits and the greedy NN behavior is made explicit so a human can verify that the mechanism is correctly implemented.