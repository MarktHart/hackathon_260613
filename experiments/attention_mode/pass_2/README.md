## What I did
This hand-built attempt implements a **minimum-KL divergence template classifier** for the five canonical attention modes (`induction`, `previous_token`, `uniform`, `sink`, `diagonal`). The classifier does not train on the data; it pre-computes ideal templates for each mode at L=64 and, for each given attention pattern, measures the Kullback–Leibler divergence of the pattern from each template. A smaller KL divergence indicates a better match, and the scores are transformed via softmax into a probability distribution over the five modes.

Key design choices:
- **Induction**: anchor to offset 8 (one of the four induction variants) as the most discriminative reference.
- **Sink**: anchor to width 4, the dominant sink size.
- **No learnable parameters**: every step is deterministic or based on a fixed Dirichlet perturbation (set in `_build_templates`). This satisfies the "hardcoded weights bonus" requirement.

Compared to the first attempt, this classifier avoids the heuristic argmax tricks that failed induction and sink modes; instead it works directly with distributional similarity, which naturally distinguishes sharply peaked (induction, previous_token, sink, diagonal) from flat (uniform) patterns.

## Why this visualisation
The Demo tab shows a single attention pattern as a lower-triangular heatmap together with its ground-truth mode label, its model-predicted mode probabilities, and a KL-divergence summary. Choosing any of the 1000 patterns from the sweep lets the grader inspect whether the heatmap structure (the location of the mass) matches the ground-truth and whether the KL-based scores assign high probability to the correct mode. The Benchmark tab loads the leaderboard of all attempts via `benchmark_panel`, showing per-mode accuracy and the headline `accuracy_canonical`.

The per-mode breakdown is the most legible signal: induction should be reliably identified because induction patterns have a single sharp peak far to the left; sink patterns also have a sharp but earlier peak; previous_token, diagonal, and uniform are easy to distinguish. Accuracy on each mode is the primary claim, and the KL visualisation lets the grader sanity-check which patterns are being confidently identified.