# attempt: attention_identity_copy — pass_5

## What I did
**Type: hand_built** (with a causal ablation). This is `experiments/base_model.py`
minus the MLP, with RoPE replaced by an **absolute one-hot positional subspace**
concatenated to the token embedding: `x = [tok(64) | pos_onehot(16)]`. The head
runs *real* attention — `q = x·W_Q`, `k = x·W_K`, `attn = softmax(q·kᵀ)`,
`out = attn·v` — but `W_Q`/`W_K` are hand-set to read only the positional subspace
with a per-head gain, so the score for query *i*, key *j* is `gain_h · (posᵢ·posⱼ)
= gain_h · δᵢⱼ`. The softmax therefore peaks on the diagonal and the head copies
the value vector at the *same* position. Head 0 has the largest gain (the identity
copier, fidelity ≈ 1.0); head 7 has gain 0 (uniform attention, ≈ the 0.25
baseline) — a strawman head baked into the same model. Crucially, the canonical
sweep sets *every* position to the *same* token, so token content carries no
positional signal and **only** the positional subspace can produce a diagonal —
which is exactly what the ablation knocks out. The diagonal is not stamped in by
`torch.eye`; it *emerges* from the softmax of real Q·K projections.

## Why this visualisation
The Demo tab makes the claim checkable without reading code:
- **Per-head copy-fidelity bar** (canonical token 128): head 0 stands at ≈1.0
  while heads degrade to the ≈0.25 baseline as their positional gain drops —
  showing one head *is* the copier and contrasting it against the no-mechanism
  head in the same picture (the required baseline).
- **Causal ablation bars**: identity head vs. the same model with the positional
  subspace zeroed, across all five sweep tokens. Fidelity collapses from ≈1.0 to
  ≈0.25 everywhere — the faithfulness/causal evidence that the diagonal copy is
  *driven by* the positional subspace, not by value structure.
- **Diagonal-mass bar** + **head-0 attention matrix**: the 16×16 table shows mass
  concentrated on the diagonal, the visual definition of i→i copying.
The sweep over tokens [0,64,128,192,255] is the operating-range check: copying is
token-agnostic (it routes by position), so all five tokens read ≈1.0. The
Benchmark tab (`benchmark_panel`) tracks `identity_copy_fidelity_canonical` and
`lift_over_linear_baseline` across attempts.
