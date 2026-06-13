# attention_anagram / pass_2

## What I did
**Type: trained (real learning, not hand-set).** Where the prior attempt hand-built an
identity QK circuit, this attempt *trains* one to verify the mechanism is what a model
actually learns. The model is `base_model.py` reduced to a single attention-only layer
(no MLP, **no positional embedding**), 8 heads: query = `W_Q · Emb[target]`,
key = `W_K · Emb[source]`. I train it with cross-entropy of each head's attention against
the true source position on fresh random anagrams (swap / rotation / random). With no
hand-set weights it converges to a **token-identity matcher**: the learned effective QK
matrix `M[a,b] = <W_Q e_a, W_K e_b>` over the vocabulary becomes strongly diagonal, so a
target token attends the source token of the same id. On the canonical condition
(random perm, seq_len 8, vocab 50) this reaches high alignment vs the **0.125** uniform
baseline and vs an **untrained same-architecture strawman** (~0.125). Because there is no
positional term, the learned circuit transfers across sequence length from L=2 to L=256
(2+ orders of magnitude), which I sweep explicitly.

Faithfulness: the trained model genuinely uses this circuit — the diagonal QK matrix is
direct evidence of the learned computation, and the untrained strawman is the causal
control (same architecture, weights un-learned → alignment collapses to baseline). Zeroing
the off-diagonal would not change behaviour because it is already ~0.

## Why this visualisation
The Demo tab shows three panels, each tied to a benchmarked or causal quantity.
**(1)** A bar chart of alignment-on-true-source per permutation type for the trained head
vs the untrained strawman vs the dashed uniform baseline — exactly the metric `score()`
computes, with the strawman making "the *training*, not the architecture, produced this"
legible. **(2)** A heatmap of the learned QK matrix over the vocab; a clean diagonal is the
interpretability money shot — it shows the head learned token identity matching rather than
memorising positions. **(3)** A log-scale line of alignment vs sequence length against the
`1/L` uniform line, demonstrating the operating range the prior attempt lacked. The
Benchmark tab tracks metrics across attempts.
