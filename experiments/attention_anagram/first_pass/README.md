# attention_anagram / first_pass

## What I did
**Type: hand_built (no training).** I express the anagram-alignment mechanism as
a single attention head with hand-set weights — `base_model.py` reduced to one
attention layer, no MLP, where `W_Q` and `W_K` are the identity acting on a
one-hot token embedding. Because the target is a permutation of the source, the
target token at position *t* is identical to the source token at its true
pre-image position, so `QK^T` produces a logit of 1 exactly at matching
(target, source) token pairs and 0 elsewhere. A temperature scales those logits
and a softmax over source positions concentrates attention on the matching
source position. The 8 heads share this circuit with a small spread of
temperatures. On the canonical condition (random permutations, seq_len 8,
vocab 50) this reaches **mean alignment ≈ 0.94** vs the **0.125** uniform
baseline (`lift ≈ 0.82`, `alignment_robustness ≈ 0.99` across swap/rotation/
random). The shortfall from 1.0 is purely token collisions: when a token repeats
in the source, identity matching cannot disambiguate which copy is the true
pre-image and splits attention — an honest, mechanism-level cap rather than
noise.

## Why this visualisation
The **Demo tab** leads with a bar chart of alignment-on-the-true-source-position
per permutation type, plotted against the dashed uniform baseline — that is
exactly the quantity the benchmark scores, so the reader checks the claim ("the
head points at the right source token") directly, and the baseline line makes
"works vs. doesn't" legible at a glance. The per-position line plot shows the
effect holds uniformly across all 8 target positions rather than being driven by
one easy slot. The **live temperature heatmap** is the causal/faithfulness knob:
red boxes mark the true source position for each target token, and sliding the
temperature down visibly smears attention off the boxes toward uniform while the
printed alignment falls toward 0.125 — demonstrating that the identity-match
logits, not some artifact, are what produce the alignment. The **Benchmark tab**
tracks these metrics across attempts.
