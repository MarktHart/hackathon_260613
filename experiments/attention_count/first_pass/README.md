# What I did

This is a **hand_built** attempt: I implemented a softmax attention readout that computes attention weights between the query and all keys, takes a weighted sum of the values, and projects the result onto the mean value direction as a proxy for the "matching value direction". The raw signal is then crudely calibrated so that at the canonical count m=4 the estimate equals 4. The hypothesis was that matching values share a common direction, so the projection magnitude should scale with the number of matches — but softmax normalisation makes the weights sum to 1, so the signal saturates. The crude rescaling is a stand-in for learning the correct gain.

# Why this visualisation

The Benchmark tab (via `benchmark_panel`) shows the full sweep metrics: exact accuracy and MAE per true count, correlation between true count and mean estimate, and robustness (min/max accuracy across counts). These directly answer the goal's question — whether the mechanism counts robustly as the count grows. The per-slice accuracy curve (count_exact_accuracy_n_<m>) is the key visual: a flat line near 1 would mean robust counting; a decaying curve reveals saturation.