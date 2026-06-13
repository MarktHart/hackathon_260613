# attention_bipartite

## What I did

Implemented a hand-built bipartite attention model that directly encodes the expected attention pattern. The model takes queries, keys, and values as input and produces attention weights that are concentrated on the diagonal blocks (within-group) and near-zero on off-diagonal blocks (cross-group). The model uses PyTorch on the GPU for accelerated computation.

The core logic:
1. For each query position, identify its group (first half = group 0, second half = group 1)
2. Force attention to be zero for all key positions in the opposite group
3. Normalise the remaining weights so they sum to 1 across each query

This approach directly implements the desired mechanism without any training or additional layers.

## Why this visualisation

The main tab in the Gradio app shows a brief explanation of the expected behavior. The Benchmark tab provides quantitative evaluation across different attempts, allowing us to see how well our hand-built mechanism performs compared to baselines and other approaches. This shows immediately whether the implemented pattern actually produces the desired attention structure. The visualisation is intentionally simple because the claim is concrete: we expect near-zero cross-group attention and high within-group attention, and the benchmark numbers directly convey this.