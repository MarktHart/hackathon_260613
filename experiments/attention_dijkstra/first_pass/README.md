


**What I did**  
This attempt implements an attention-style mechanism that computes single-source shortest-path distances using iterative relaxation. The model function takes in an adjacency matrix and a source node, initializes distances as infinity (except zero at the source), then performs 10 hops of soft-min aggregation via Torch on the GPU. In each hop, every node $ v $ updates its distance as the pointwise minimum over all $ d[u] + \text{weight}(u,v) $, approximating theBellman–Ford / Dijkstra propagation. No learned weights or parameters are used; the entire circuit is hand-set as matrix and tensor operations.

**Why this visualisation**  
The `Demo` tab intentionally remains a stub — with a hand-built non-parametric mechanism, the interesting story is not interactive exploration but comparison against the one-hop baseline and against the true Dijkstra distances. Hence the visualisation focuses on the `Benchmark` tab, which shows the headline metric `dijkstra_robustness` and the canonical `distance_accuracy`. Future attempts with learned heads can add an interactive graph view; this first pass establishes a clean, interpretable baseline for the pipeline leaderboard.