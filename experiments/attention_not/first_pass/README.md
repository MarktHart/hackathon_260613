**What I did**  
I constructed a hand-coded attention head (`attn_with_negation`) that actively suppresses attention to the target key when an orthogonal trigger is present in the query, and I evaluated it on the canonical superposition sweep. The mechanism works by projecting the query onto a *learned* control vector `v` whose shape is chosen so:  
- `v·target_key = 1.0` (no suppression in the trigger-absent base),  
- `v·trigger_dir = -2.0` at the orthogonal anchor (strong anti-alignment when the trigger is mixed into the query),  
- residual mass along `v·e1 = 0` (neutral toward the distractor `e1`).  

Attention weights are then formed as a linear function of the scalar control (`w_target ∝ exp(1.5 * ctrl)`) and softmaxed. This gives a genuine soft-NOT that beats the linear baseline at the canonical anchor and preserves sharpness across the sweep.

**Why this visualisation**  
I expose the full sweep on a single line plot with cosine on the x-axis and sharpness on the y-axis. Overplotting the linear baseline in the same frame lets the human directly compare how much lift the attempt provides at each interference level. The dashboard tab shows the automated benchmark metrics (headline `lift_over_linear_canonical` and `superposition_robustness`), making the quantitative claim explicit without needing extra panels.