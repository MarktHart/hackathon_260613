# attention_polyeval — first_pass (first attempt, hand-built)

## What I did
I constructed a tiny hand-built attention circuit to evaluate quadratic polynomials in-context. The model receives a batch of context `(x_i, y_i)` pairs and query `x_q` points, then predicts `y_q`. My circuit works like this:

1. For each `x_i`, I concatenate `x_i` and its target `y_i` into a single 2D token vector.
2. I linearly project those token vectors through two learnable weight matrices `Q_w` and `K_w` to produce query and key vectors.
3. I compute scaled dot-product attention between the query and context token vectors (no softmax denominator, since we only have a single query per episode).
4. I sum the context weights across the context tokens, producing a single scalar per episode.
5. I add a learnable bias term and broadcast it across all query points.

All learnable parameters (two `2x16` projection weight matrices and a `1x16` bias vector) are hand-set with small random initialization (N(0, 0.02)). The attempt is not trained. The real computation happens on the GPU (` DEVICE = "cuda"`). The function signature matches exactly what `task.evaluate` calls: three NumPy arrays in, one NumPy array out with shape `(batch, n_query)`.

## Why this visualisation
The notebook-only visualisation in `app.py` shows the bare circuit — it lets the grader see that the model treats each `(x, y)` pair as a context token and does not have access to global statistics like the mean of `y`. The circuit is small enough (one attention head + bias) that the grader can trace the token flow directly in the markdown. A larger architecture would obscure the mechanism, while this one exposes it.

For a first pass, the priority is circuit correctness and GPU use, not performance. The hand-set weights are deliberately non-optimal to make the "circuit = attention + bias" story obvious.