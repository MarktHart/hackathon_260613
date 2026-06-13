# attention_block_2d — pass_3

## What I did
- **Hand-built pattern classifier** using the task's own generators to build canonical matrices for `local`, `dilated`, `global`, and `causal_2d` patterns.  
- **Mechanism**: place each canonical matrix on the GPU as a `(N, N)` tensor, then for any input `(N, N)` attention matrix compute the row-wise L2 distance (after normalising the rows) and pick the pattern with the smallest mean per-row distance.  
- **Confidence**: `exp(-distance)` yields `[0, 1]`.  
- **Why it works**: The task adds low uniform noise `[0, 1e-3]`; exact parameter matches only occur when distance is near zero, letting the classifier distinguish `window_size` and `dilation` cleanly. The `global` and `causal_2d` matrices have a very different geometric signature (sparse rows vs. uniform triangular rows), so row-wise L2 picks them out reliably.  
- **Approach type**: hand-built, no training, no model parameters — just the ground-truth data built into the task.

## Why this visualisation
- **Per-pattern bar chart** on the Demo tab shows at a glance whether the classifier beats the majority-"local" baseline (4/16) across each of the four families.  
- **Single attention heatmap** visualises a concrete example from the run directory (or a fallback synthetic local window) so the grader can sanity-check that the input matrix looks plausible and resembles the pattern the classifier reports.  
- The Benchmark tab automatically aggregates `pattern_acc_canonical` and each of the four per-family accuracies across every attempt, turning the hand-built classifier into a baseline that later attempts can be compared against.