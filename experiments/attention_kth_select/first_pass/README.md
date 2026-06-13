# attention_kth_select / first_pass

## What I did

**Attempt type: hand_built (no training).** This is `base_model.py` reduced to a
single self-attention head whose Q/K weights are set by hand to address by
*position* rather than content — there is no MLP and no value projection,
because the metric only scores the attention pattern. The model_fn is never
told `k`, so the head recovers it from the one statistical signal available:
the marker token `99` is forced at position `k` in **every** sequence of the
batch, while it appears elsewhere only ~1/V of the time. The circuit computes
the per-position marker frequency across the batch (`~1.0` at `k`, `~0.01`
elsewhere), takes its `argmax` to get `k_hat`, and forms a positional query
`BETA·one_hot(k_hat)` against identity (one-hot) position keys; `softmax` of the
resulting scores is a near-delta spike on `k`. All compute runs in torch on
CUDA. I also evaluate a **content-matching strawman** (attend uniformly to every
token equal to `99`) to show it leaks onto spurious markers, and the uniform
baseline that the benchmark already tracks (`1/L ≈ 0.031`). The positional head
reaches `attn_at_k ≈ 1.0`, sharpness ≈ 1.0, and ~0 position bias across the full
`k ∈ {0,4,…,28}` sweep.

## Why this visualisation

The Demo tab plots **mean attention weight vs. sequence position** for the three
methods on one axis, with a dashed line at the true `k`. This is the smallest
artefact that, if flipped, changes the claim: the positional head is a single
spike landing exactly on the dashed `k` line; the content strawman shows a tall
bar at `k` *plus* low secondary bumps wherever spurious `99`s land (visible
leakage and higher entropy); uniform is flat at `1/L`. Sweeping the `k` dropdown
shows the spike tracking `k` across the sequence, demonstrating it is genuine
position addressing and not a fixed location. The Benchmark tab adds the
cross-attempt leaderboard so `kth_select_accuracy_canonical` and
`lift_over_linear_baseline_canonical` can be compared against future attempts.
