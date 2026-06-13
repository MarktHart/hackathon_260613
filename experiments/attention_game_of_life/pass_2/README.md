# What I did

This is a **hand_built / interp** attempt that expresses one step of Conway's
Game of Life as the **smallest delta from `base_model.py`**: token embedding →
**one self-attention block** → **one MLP** → residual → unembed. Each cell is a
token (sequence length `N = H*W`); the attention layer uses a hand-set
relative-position bias (a toroidal neighbour mask) so every query cell attends
*uniformly* — weight `1/8` — to its eight neighbours, with `V` reading the
alive bit, making the attention output literally `neighbour_count / 8`. The
output projection lifts that to the integer count `n`, which the residual
stream carries beside the cell's own state `s`, and a tiny hand-set ReLU MLP
applies the rule `alive_next = (n == 3) OR (n + s == 3)` (two triangular
"exactly-k" detectors). **Every weight is set by hand — no training, no conv —
so the mechanism is fully transparent and is genuinely *attention* doing the
neighbour-counting.** Faithfulness is checked causally: `main.py` re-runs the
same circuit with the neighbour mask replaced by global uniform attention
(`ablate`) and with the attention output zeroed (`selfonly`), and records that
F1 collapses in both, while the full circuit scores `1.0` and a grid-size sweep
(8→64) shows it holds across ~2 orders of magnitude of cell count.

# Why this visualisation

The Demo shows five panels on one board: current state, the **independent
NumPy** ground-truth next state, the **circuit's** prediction, and a match map
— crucially the truth and the prediction come from *different code paths*
(NumPy reference vs. the torch attention circuit), so an all-green match map is
a real correctness check, not the self-comparison the previous attempt was
faulted for. The attention panel renders the softmax weights of a chosen query
cell, letting a viewer confirm by eye that the head lights up *exactly* its 8
toroidal neighbours at `1/8` each — i.e. that the mechanism really is attention
gathering neighbours, not a hidden convolution. The ablation table puts the
headline claim next to its falsifier: full neighbour-attention F1 vs. global
attention vs. self-only vs. the static copy baseline, so the causal necessity
of the neighbour mask is one glance. The operating-range table sweeps grid size
to show the hand-set circuit generalises rather than overfitting the 16×16 case.
