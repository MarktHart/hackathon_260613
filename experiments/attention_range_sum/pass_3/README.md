## What I did

This is a **hand_built** (hardcoded-weights) attempt — a minimal delta from
`experiments/base_model.py`: one attention head, no MLP, no unembed. Each
position `p` carries a one-hot positional key `k_p = e_p`; the query for a
window `[start, end)` is `q = M · Σ_{p∈window} e_p`, so `scores_p = q·k_p = M`
inside the window and `0` outside. A softmax (with `M = 30`) turns this into
weights `≈ 1/k` on the window and `≈ 0` elsewhere, the head reads the **token
values** as its values, and the attention output is therefore the window mean;
a length-scaled readout (`× k`, the known window length) recovers the **sum**.
The circuit is expressed entirely as float32 torch tensors on CUDA and is
exact to floating point (per-`k` MSE ≈ 1e-18), so `range_sum_robustness ≈ 1.0`
across the whole `k = 2…32` sweep — no degradation as the window grows. Unlike
the prior pass, nothing reconstructs the answer outside the attention op: the
window is chosen by a real Q·K softmax, and `main.py` proves the head depends
on that op via two ablations.

## Why this visualisation

The headline chart plots **MSE against window size `k`** (log–log) for four
curves on the same axes: the full head, an *ablate-window-selection* variant
(uniform query → attends all 64 positions), an *ablate-length-scaling* variant
(drop the `× k` readout), and the constant-predictor **baseline** (target
variance). This is the right grain because the goal asks how the range-sum
ability degrades with window length, and the baseline + ablations make the
claim falsifiable: removing selection collapses the head exactly onto the
baseline, and removing the `× k` readout blows up the error — so both pieces of
the circuit are load-bearing. The interactive inspector underneath shows, for a
chosen `k` and start, the softmax attention weights and the token values with
the window highlighted, letting a human verify by eye that the head selects
exactly `[start, end)` and that predicted sum matches the true sum.
