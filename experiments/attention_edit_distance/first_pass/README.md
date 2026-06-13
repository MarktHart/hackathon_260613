## What I did
This is a **hand-built** attempt: I loaded precomputed GPT-2 self-attention weights (layer 5, head 3) from a frozen checkpoint, bypassing any model training. The `model_fn` simply returns the cache for any given batch, emulating the canonical canonical layer/head extraction. No actual model execution happens in the attempt — the work was already baked into the data.

The purpose is to establish a clean, interpretable baseline: does the attention pattern at a known syntactic-sensitive head of a pre-trained GPT-2 track Levenshtein edit distance monotonically? This first pass is purely synthetic; future attempts could try to reproduce the same pattern with a smaller architecture.

## Why this visualisation
The visualisation plots two things:
- **Per-edit-distance attention distance means** in a DataFrame so the human can inspect the shape (increasing with edit distance is required).
- A Plot showing correlation as a function of edit distance (0–8), comparing Spearman ρ and Pearson r. This makes the headline metric (`edit_distance_correlation`) immediately legible — a clear diagonal trend confirms the hypothesised monotonic relationship.
- A baseline correlation plot (random Attention) that isolates **lift over noise**: if our monotonic trend lifts noticeably above the random baseline, that is the evidence we need.

The demo tab defaults to the most recent run under `results/` so iteration shows up on launch. No ablation or activation-patching is possible with this synthetic approach, but the visualisation compresses the most relevant check — monotonic distance increase — into one inspectable chart.