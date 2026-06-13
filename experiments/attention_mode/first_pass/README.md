## What I did
Hand-built a simple heuristic `model_fn` that inspects the structure of each attention pattern and assigns a "logit" score to each of the 5 modes based on crude rules:
- **induction**: how concentrated the most common peak column is across rows.
- **previous_token**: proportion of rows where the maximum weight falls on the column directly left of the row.
- **uniform**: 1 minus the row-wise standard deviation (higher score = more uniform row).
- **copying**: placeholder heuristic (set to mid-value for demonstration; see notes).
- **first_token**: proportion of rows where the maximum weight falls on column 0.

The scores are rescaled to a [-5, 5] range to resemble logits so a simple argmax works. This is a **first-pass** attempt to validate that the task can be solved without learned weights — we can later add training.

## Why this visualisation
The Demo tab shows how one random pattern from a chosen sweep point looks, what the ground-truth mode is, what the heuristic classifies it as, and the raw heatmap of the attention matrix. This makes it easy to see: (1) whether the heuristic correctly captures the mode’s structural signature, and (2) how robust that visual pattern becomes as we lower **α** (noisier patterns).

The Benchmark tab loads `agentic.experiments.benchmark_panel` and shows all attempts at the `attention_mode` goal in a leaderboard with accuracy per α and the robustness ratio — the main metric this task cares about.