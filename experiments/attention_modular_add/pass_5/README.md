# pass_5 — Hand-built Fourier attention head

## What I did
This is a **hand_built** attempt (no training): a single attention head's Q/K
projections are set by hand to the canonical modular-addition circuit from Nanda
et al. (2023). The query at position `a` carries `[sin(2πk a/p), cos(2πk a/p)]`
for every frequency `k = 1..48`, interleaved across the 128 channels; the key at
position `b` carries the **conjugate** `[-sin(2πk b/p), cos(2πk b/p)]` on the
*same* channels. Because each frequency lives on a shared channel pair for both
Q and K, the regressed Q-side and K-side freq-`k` subspaces coincide, giving
alignment = 1.0 across all 48 frequencies, with
`q·k = Σₖ cos(2πk(a+b)/p)` depending **purely** on `a+b` — verified numerically:
for every `a`, `argmax_b q·k` lands exactly on `a+b ≡ 0 (mod p)`. Versus the
`random_model_fn` strawman (alignment ≈ `2/d_head` = 0.0156), this lifts the
headline `fourier_alignment_canonical` to a perfect 1.0 (lift 0.984). One honest
caveat: the benchmark's `phase_error` is π/2 for this exact conjugate head, not 0
— its formula `|angle(w_Q·conj(w_K))|` is minimised when `w_Q ≈ w_K` (the `a−b`
pattern), whereas the literal conjugate `w_K = conj(w_Q)` that produces the true
`a+b` peak sits at π/2 by construction. I kept the faithful `a+b` circuit rather
than game that secondary metric. The model is expressed entirely as torch tensors on
`cuda` and is fully vectorized — fixing the timeout in pass_4, whose Python-loop
basis build and per-element GPU slice-assign were too slow. (Faithfulness note:
this is a synthetic hand-set head, so there is no trained model to ablate; a
causal check would zero each frequency's channel pair and confirm the attention
peak at `a+b` flattens.)

## Why this visualisation
The Demo shows the two halves of the claim directly. The bar chart of
per-frequency `cos(2πk(a+b)/p)` makes visible that *every* frequency contributes
coherently near the answer — the reason mean alignment is high rather than
concentrated on one frequency. The line plot sums those contributions over all
candidate sums `s ∈ [0,p)` and shows a single sharp peak exactly at
`(a+b) mod p`, the behavioural payoff of the Fourier mechanism. The Benchmark tab
drops in the shared panel so this attempt's `fourier_alignment_canonical` is
comparable against every other attempt.
