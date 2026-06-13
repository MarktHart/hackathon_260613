# attention_cyk — pass_2

## What I did
This is a **hand_built / interp** attempt: a genuine two-head attention circuit,
not a heuristic loop. It is `base_model.py` minus the MLP, with two hand-set
attention heads on the sign-embedding (`(`→+1, `)`→−1). **Head A** is a causal
counting attention (strict lower-triangular pattern, value = sign) whose output
at position `p` is the prefix bracket depth `D(p)`. **Head B** is a
depth-matching attention: its score for split `k` is `-(D(k) − D(i))²`, which is
exactly a linear QK dot product `⟨q, φ(k)⟩` with `q = [2·D(i), −1]` and
`φ(k) = [D(k), D(k)²]`; a softmax over the cell interior concentrates mass on the
*balance points* `D(k) = D(i)` — precisely the `S→S S` / `X→S R` CYK splits. A
small gate routes mass to `k = i+1` for pure-wrap cells `(W)` where `S→L X` is
the only firing production. All compute runs in torch on `cuda`. The circuit
reaches ~1.0 split accuracy across every span length (lift ≈ +0.6 over the
uniform baseline, robustness ≈ 1.0), and as a **faithfulness/causal check** I
ablate head A (zero the depth channel): the circuit then collapses to the
uniform baseline, proving the depth-matching head is what does the work. The
depth equality `D(k)=D(i)` is also exactly what a *trained* attention head would
have to learn, so the analogous check there is activation-patching the depth
feature.

## Why this visualisation
The Demo overlays the two things the claim depends on. The top panel is the
string's bracket-depth profile with a dashed line at the cell's start depth
`D(i)`; the bottom panel is the head's actual attention distribution over
candidate split points, with CYK-correct splits coloured green. Putting them on
shared split-position axes lets a human verify the mechanism directly: attention
spikes exactly where the depth curve returns to `D(i)`, and those spikes are the
green (correct) bars — the visual *is* the argument that the head implements the
split rule. The companion bars give the headline (`full` vs `uniform`) plus the
**ablation** (`depth-ablated` falling back to uniform), and the per-span line
shows the result holds from span 3 up to 9 rather than only on a cherry-picked
example.
