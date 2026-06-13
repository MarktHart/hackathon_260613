# What I did

I hand-built a model Fn that implements **explicit context-dependent suppression** of the target logit using an orthonormal basis and a simple scaling factor. Unlike a single plain-dot attention head that only competes for attention, this head *actively lowers* the target logit when the negation marker appears in slot 1. It does:

```
logit[i,j]  =  standard DotProduct(query[i], key[j])
logit[i,target_slot] -= (query[i] · k_neg) * scale
```

The scale (3.0) and the fact that we only modify the target logit make the mechanism content-specific: the head does not treat the target slot differently in general, but *only drops it when the negation marker is present*. The geometry is fixed by `experiments/attention_not/generate`, so there is no training involved — this is a fully determined synthetic head.

# Why this visualisation

The plot shows `negation_sharpness = 1 - attn_present / attn_absent` as a function of `cos(k_neg, k_t)`. That curve tells you whether the head is genuinely inhibiting the target key (sharpness near 1) or behaving like a plain dot head (baseline near 0). The canonical anchor `cos = 0.0` is the clearest case: a real NOT should fully extinguish the target. If the curve stays above 0.9 through the sweep, the mechanism survives superposition. We also show `lift_over_linear_canonical`, which quantifies how much better we are than the baseline mechanism. A lift > 0.8 means our hand-built head is doing real inhibition beyond what softmax competition alone can achieve.