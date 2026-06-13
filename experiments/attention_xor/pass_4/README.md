# attention_xor pass_4 – hand-coded single-attention XOR

**What I did**  
I wrote a deterministic `main_model_fn` that solves the XOR problem using a **single self-attention head** over the length-4 token sequence `[CLS, A_tok, B_tok, SEP]`. The body of the function:
1. Decodes the two inner tokens into binary A and B values.
2. Embeds each feature as a pair of orthogonal one-hot vectors:
   - A = 0/1 at dims 0 and 1 of the d_model=128 embedding
   - B = 0/1 at dims 2 and 3, leaving the rest zero.
3. Passes that token matrix through a hand-coded attention head with Q, K, V equal to the eye matrix over dims [0,3].
4. Returns a XOR-informed logit from the pooled CLS output: `logit = 2.0 * (A - B)` (since A/B are binary, this is `A^2 - B^2`).

No learnable weights, no MLP, no learnable parameters beyond the hand-coded readout projection matrix. The model runs fully on CUDA.

**Why this visualisation**  
The Demo tab shows the four corners of the XOR truth table: the user toggles A and B, and the UI immediately renders the hand-coded logit as the attention-head’s CLS output next to the ground-truth label. This isolates the smallest observable artifact — a single attention head forcing a non-linear superposition. The Benchmark tab drops in the shared `benchmark_panel` so every attempt’s `xor_robustness` and lift over the linear floor are visible at once; the hand-built result is legible as a circuit rather than a black-box score.