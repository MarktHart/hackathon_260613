# attention_reverse / first_pass

## What I did

This is a **hand_built** attempt: a single attention head with **no learned
weights** that implements exact sequence reversal. It is `base_model.py` minus
the MLP and causal mask, with RoPE replaced by an *exact discrete-Fourier mirror
encoding*. For a length-`L` slice I build position features
`phi(p) = [cos(2πmp/L), sin(2πmp/L)]` over frequencies `m = 0..L-1`; keys use
`phi(j)` and queries use `phi(L-1-i)`. Then `q_i·k_j = Σ_m cos(2πm((L-1-i)-j)/L)`
equals the discrete delta kernel — `L` exactly at `j = L-1-i` and `0` everywhere
else — so a plain softmax yields a near one-hot attention pattern on the mirror
position. Value vectors are one-hot token embeddings, so the attended output at
position `i` is the one-hot of the token at `L-1-i`, returned directly as logits.
Because the frequencies are read off the actual input length `L`, the head is
parametric in length and extrapolates perfectly: **accuracy 1.000 and mirror
attention mass ≈1.000 at lengths 8, 16, 32, 64**, giving headline
`length_generalization_robustness = 1.0` (vs. random baseline 1/16). The
contrast to a length-16 lookup is exactly what the metric isolates — this head
has nothing to memorise, so robustness is maximal rather than ~0.

A causal model is the natural strawman (position 0 must read the future), and the
benchmark's `random_baseline_accuracy` (0.0625) and data-driven
`identity_baseline_accuracy` (no-reversal) sit far below this head. *Faithfulness
note:* this is a synthetic hand-set circuit, not a probe of a trained model, so
there is no ablation of a learned weight — but the mechanism is itself the
ablation target: zeroing the Fourier query/key offset (querying `phi(i)` instead
of `phi(L-1-i)`) collapses the anti-diagonal to the main diagonal and accuracy
drops to the identity baseline.

## Why this visualisation

The Demo tab plots the head's `(query × key)` attention matrix with the mirror
line `j = L-1-i` overlaid. The single claim — "this head sends query `i` to key
`L-1-i`" — is true iff the attention mass lies on a clean **anti-diagonal**
tracing that overlay; any smear or wrong diagonal would be visible immediately.
The length dropdown lets you confirm the anti-diagonal stays sharp as `L` grows
to 64 (the generalisation test), and the reported accuracy / mirror-mass put a
number on what the heatmap shows. The temperature slider demonstrates the
mechanism is robust: even at low `beta` the delta kernel keeps mass on the
mirror. The Benchmark tab tracks `length_generalization_robustness` and
per-length accuracy across attempts so any future learned attempt is compared on
the same yardstick.
