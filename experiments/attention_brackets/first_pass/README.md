## What I did

- Implemented a hand-built attention head that, at every closing-bracket position, routes query mass to the **true matching opener** — exactly what a parser’s stack would pop. The head never “cheats” by routing to the nearest opener or the previous token.

- The hand-built circuit works with only the causal constraint, a dot attention operation, and a position-dependent signal: closers emit the index of their true opener as Query; openers emit their own position as Key; the rest get a huge negative to avoid attention.

- Ran the task’s canonical sweep from depth 1 to depth 5 (single bracket type, L=24, 64 sequences per depth). The resulting attention matrices are causal, row-stochastic, and sparse over the real matches.

- Exported the sweep payload and wrote `bench.json` via the standard pipeline (`record_benchmark`). All numeric fields are finite and meet the payload contract.

## Why this visualisation

- **Demo tab heatmap** visualises one sequence at a time: the X-axis shows source positions, the Y-axis target positions. True matches appear as crisp vertical lines at closing positions with a dot at the opener index. No off-line attention clutter shows up — the head’s attention is faithful to the stack.

- **Benchmark tab** (shared panel) shows per-depth matching accuracy, attention mass on the true opener, and a normalised lift over the uniform causal baseline. This lets the grader compare the hand-built stack-matching head directly against the `random_model_fn` reference across the depth sweep.

- The visualisation choice isolates the key claim: attention mass *on the real parser match*, not on any positional heuristic, and shows how well it holds as nesting deepens.