## What I did

This is a **hand_built** attempt. The mechanism is `base_model.py`'s
dot-product attention with exactly one delta: the score `K @ q` is multiplied
by a scalar **temperature** `beta` before the softmax (base_model uses a fixed
`1/sqrt(d)` scale; I make that scale a settable parameter and turn it *up*).
The first_pass used raw `K @ q` (β=1) and scored only ~0.19 target mass —
the softmax over L=12 keys was simply too soft. With β=16 the target, which is
the unique argmax for every interference level below cos=1 (the target key
equals the query, logit 1, while the distractor logit is its cosine), captures
**~1.0** of the mass: canonical alignment jumps from 0.19 to ≈0.99. The run
also builds three explicit strawmen under the identical condition — raw `K@q`
(β=1), a **temperature=0 ablation** (logits→0, collapses to the uniform head),
and random logits — saved for the Demo. The ablation is the faithfulness/causal
check: zeroing the temperature knocks the retrieval behaviour out completely,
back to uniform 1/L, showing the alignment is caused by the tempered `K@q`
circuit and nothing else. (This is a synthetic hand-set circuit, not a trained
model; a trained-model version of the same check would be ablating the head's
logit-scale parameter.) Note the headline robustness is ≈0.5 by design, not
weakness: at cos=1 the distractor key is *mathematically identical* to the
target, so attention mass can at best split 50/50 — 0.5 is the physical
ceiling, and the first_pass's higher 0.889 was an artifact of a flat, near-chance
curve.

## Why this visualisation

The Demo's main panel puts **global alignment (mass on the target)** on the
y-axis against the **interference axis (distractor cosine)** on the x-axis, with
all four mechanisms and the uniform 1/L baseline on one set of axes. That is
exactly what the goal asks — "how much alignment survives as the distractor
approaches the target" — and the strawmen make the claim testable: the tempered
curve sits near 1.0 and only falls to the 0.5 ceiling line at cos=1, while raw
`K@q` hugs ~0.19 and the temperature=0 and random heads sit on the uniform
baseline. The second panel is a 3-bar mass split (target / distractor / other
keys) at a slider-selected slice, so a human can watch the distractor steal mass
only as cos→1 and confirm the 50/50 tie at maximum interference rather than a
collapse onto the wrong key. The dashed 0.5 line annotates the ceiling so the
robustness number reads as "at the physical limit," not "broken."
