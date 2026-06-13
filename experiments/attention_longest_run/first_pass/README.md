# attention_longest_run — first_pass

## What I did
This is a **hand-built** measurement circuit (no training). The task asks: given
noisy per-head attention weights, recover the longest contiguous run of
positions a head attends to above the 0.5 threshold. The circuit is the minimal
delta from `base_model.py`'s attention readout — it operates directly on the
attention-weight tensor that a head produces: (1) binarize at the canonical
threshold (`weights > 0.5`), (2) apply one **morphological closing** (dilate→erode,
kernel 3) along the sequence axis to repair single-position drop-outs caused by
the `N(0,0.15)` noise, and (3) take the longest contiguous run via a sequential
run-length scan. All three steps run as torch ops on CUDA (`max_pool1d` for the
morphology, a 64-step GPU scan for the run length). The closing step is the whole
hypothesis: the dominant error of naive thresholding is *under*-counting because
one in ~20 high-weight positions noise-flips below 0.5 and splits a run; closing
bridges those single gaps. I report the denoised circuit against two strawmen
measured under the identical condition — raw threshold (no closing) and a
predict-the-mean baseline (which `benchmark.py` also computes).

## Why this visualisation
The **Demo** bar plot shows one head's raw attention weights with the red 0.5
threshold line, the orange band marking the true implanted run, and bars colored
by what the denoised mask keeps — so you can see directly that the detected run
matches the implanted span and read the predicted vs. true `L` in the title.
Sweeping the `L` and head dropdowns (heads cycle difficulty d=0.3/0.5/0.7/0.9)
lets you confirm the method holds from `L=1` to `L=16` and degrades gracefully
as difficulty drops. The **summary** bar chart puts MAE on the y-axis grouped by
difficulty, with raw-threshold vs. denoised side-by-side and the baseline as a
dashed reference — the single comparison that proves the closing step earns its
keep and that both beat predicting the mean. Run-length and difficulty are the
two axes the goal's metrics hinge on, so they are exactly the chart's axes.
