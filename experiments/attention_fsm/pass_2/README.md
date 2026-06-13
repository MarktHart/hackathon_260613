# What I did

This is a **hand_built** interpretability attempt (no training, hand-set
weights). I first proved the goal's DFA is a **Z/3 permutation automaton**:
every token is a bijection on states, and `state[t] = (s0 + Σ_{i≤t} inc(token_i))
mod 3` with `inc = [0,1,2,1]` (asserted in `main.py`). The mechanism the goal
asks about — sequential state tracking — is therefore a *prefix sum of group
increments*, which a **single causal attention head** computes exactly: pattern
= `tril(ones)` ("attend to every earlier token"), value = the token's increment,
read out off a 3-phase representation. The model is `base_model.py` minus the MLP
with hand-set weights: token embedding → increment, one head → prefix sum, a
boundary embedding injecting `s0`, phase unembed → `(s0+S_t) mod 3`. Because a
permutation automaton never synchronizes, `s0` is provably *unidentifiable from
tokens*, so it is supplied as the one-integer-per-sequence boundary condition —
but **ablating the head collapses accuracy from 1.00 to chance (0.33)**, proving
the head (not the anchor) does the tracking. The circuit is exact across seeds
and from length 8 to 1024 (≥2 orders of magnitude); all compute runs in torch on
`cuda`. Unlike the prior `first_pass` oracle, this never reads the answer key —
it reads only the initial condition and *computes* every subsequent state from
tokens.

# Why this visualisation

The headline bar chart is the falsifiable claim: **full head = 1.00, head
ablated = chance, random = chance**, all under the canonical seed-0 condition —
"the attention head is the tracker" survives only if the middle bar drops, and it
does. The per-position line shows accuracy stays flat with depth (no drift), and
the operating-range plot (log-x, L=8→1024) shows the same circuit is exact far
outside the demo regime, distinguishing a real mechanism from memorization. The
`tril(ones)` heatmap makes the attention pattern itself legible, and the
per-sequence trace table exposes the arithmetic step by step (token → increment →
true state → full pred ✓ → ablated pred ✗) so a human can hand-check that the
ablated column only matches when the running sum happens to be 0 mod 3.
