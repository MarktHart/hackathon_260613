# Neural-Network Building Blocks

A roughly-sorted-by-complexity list of classical algorithms that a small
transformer (or comparable architecture) should be able to learn end-to-end.
Each block is something we can train, probe, and ideally interpret
mechanistically. Higher tiers compose lower ones — dependencies are noted in
`builds on:` lines.

Conventions:
- **I/O** sketches the task as (input → output).
- **Why interesting** notes the mechanism we expect to find inside the model.
- Within a tier, items are also roughly ordered by complexity.

---

## Tier 0 — Boolean & arithmetic primitives

These are the things a single attention head or MLP neuron should be able
to express. If we cannot find these cleanly, nothing higher will be clean either.

1. **AND / OR / NOT over two tokens**
   - I/O: two indicator features → one indicator.
   - Why interesting: minimal attention pattern — query selects token A, key
     gates on token B, value is constant. Probe: does a single head implement it?

2. **XOR / parity over two tokens**
   - I/O: two bits → one bit.
   - Why interesting: not linearly separable, forces a 2-layer construction or
     a quadratic interaction inside an MLP. Cleanest test for "MLPs do
     nonlinearity, attention does routing".
   - builds on: AND/OR.

3. **Equality check (token == token)**
   - I/O: (x, y) → 1[x = y].
   - Why interesting: prerequisite for induction, copy, lookup. Q·K dot
     product peaks on identity.

4. **Linear sum / dot product over a window**
   - I/O: vector of k scalars → sum.
   - Why interesting: pure attention-as-averaging; the weights *are* the
     algorithm.

5. **Modular addition (a + b) mod p**
   - I/O: (a, b) → (a+b) mod p.
   - Why interesting: the canonical grokking task; circular/Fourier structure
     in embeddings is interpretable in closed form.

6. **One-hot lookup / dispatch**
   - I/O: (key, table) → table[key].
   - Why interesting: tests whether attention learns key-conditioned routing
     without leaking through the value path.

---

## Tier 1 — Single-axis positional patterns

Algorithms whose only complexity is "where to look". The content function is
trivial; the positional / relative-distance reasoning is the whole point.

7. **Identity copy (token at i → output at i)**
   - I/O: sequence → same sequence.
   - Why interesting: degenerate baseline; confirms residual stream + value
     path actually carry information.

8. **Shift-by-k (output[i] = input[i − k])**
   - I/O: sequence, fixed k → shifted sequence.
   - Why interesting: tests learned relative position. With RoPE, this should
     correspond to a fixed rotation in the QK plane.
   - builds on: equality.

9. **Previous-token head**
   - I/O: at each position, copy the previous token's embedding.
   - Why interesting: a known transformer primitive; the building block for
     induction heads.

10. **Distance / magnitude comparison (|i − j| < r)**
    - I/O: pair of positions → 1 if within radius r.
    - Why interesting: RoPE-like distance metric; combined with AND from
      Tier 0 gives "attend to neighbour AND same token".
    - builds on: AND, shift-by-k.

11. **Argmax / argmin over a window**
    - I/O: sequence of scalars → index of extremum.
    - Why interesting: softmax-as-argmax in the limit; easy to probe via
      attention entropy.

12. **Counting (how many times does X occur in context)**
    - I/O: sequence, query token → count.
    - Why interesting: requires summing many keys; tests whether the model
      uses a "uniform attention" trick or learns position-invariant tallying.

---

## Tier 2 — Span & region attention

Algorithms that pick out a contiguous block — the natural next step once you
can both *select content* and *reason about position*.

13. **Boundary detection (start-of-span, end-of-span markers)**
    - I/O: sequence with delimiters → indicator at each boundary.
    - Why interesting: precondition for any span task; usually one head per
      boundary type.
    - builds on: equality, previous-token.

14. **Attend to (start, end) span**
    - I/O: (start_idx, end_idx, sequence) → pooled span content.
    - Why interesting: combines distance-comparison and boundary detection;
      "MASK + RoPE-like distance" lives here.
    - builds on: distance comparison, boundary detection.

15. **Bracket / parenthesis matching (depth tracking)**
    - I/O: sequence of `( )` → depth at each position, or matching index.
    - Why interesting: requires running a counter — first place where a
      naïve attention pattern fails and the model needs MLP state.
    - builds on: boundary detection, counting.

16. **2-D (a, b, c, d) block attention (image / spectrogram patches)**
    - I/O: (top, bottom, left, right, grid) → pooled block.
    - Why interesting: factorises into two 1-D span attentions; tests whether
      heads specialise per axis.
    - builds on: span attention.

17. **Hierarchical pooling (window → segment → sequence)**
    - I/O: sequence → multi-scale summary.
    - Why interesting: stress-tests the residual stream's ability to carry
      summaries at multiple granularities.
    - builds on: span attention.

---

## Tier 3 — Multi-token composed patterns

These compose Tier-1/2 primitives into the workhorse circuits we already see
in real LMs.

18. **Induction head (A B … A → B)**
    - I/O: sequence with a repeated bigram → predict the second occurrence's
      continuation.
    - Why interesting: the canonical 2-head circuit (previous-token →
      key-by-content). Reference target for any mechanistic story.
    - builds on: previous-token, equality.

19. **N-gram match with skip / wildcard**
    - I/O: pattern with one wildcard → matches in sequence.
    - Why interesting: tests whether induction generalises beyond bigrams.
    - builds on: induction.

20. **Sequence reversal**
    - I/O: x1…xn → xn…x1.
    - Why interesting: requires position-i to attend to position-(n−i),
      which is a learned reflection — non-trivial under causal masking.
    - builds on: distance comparison.

21. **Set deduplication / unique tokens**
    - I/O: sequence → first occurrences only.
    - Why interesting: "have I seen this before?" — combines equality with a
      running OR over history.
    - builds on: equality, OR.

22. **Sorting (small k)**
    - I/O: k scalars → sorted k scalars.
    - Why interesting: easy to express as "for each rank, argmin of remaining";
      tests whether multi-head attention learns per-rank specialisation.
    - builds on: argmin, counting.

23. **Histogram / bag-of-counts**
    - I/O: sequence → vector of counts per vocab item.
    - Why interesting: pure permutation-invariant readout.
    - builds on: counting.

---

## Tier 4 — Stateful / recursive logic

The model now needs to carry running state across positions. Pure attention is
no longer enough; the MLP+residual stream must implement a register.

24. **Integer addition with carry**
    - I/O: two digit sequences → their sum.
    - Why interesting: carry propagation is the simplest learned recurrence;
      classic source of length-generalisation failure.
    - builds on: modular addition, previous-token.

25. **Finite-state machine simulation (regular language acceptance)**
    - I/O: string, FSM spec → accept/reject.
    - Why interesting: tests whether the residual stream encodes a small
      discrete state.
    - builds on: lookup, equality.

26. **Dyck-k language (nested brackets of k types)**
    - I/O: string of brackets → well-formed?.
    - Why interesting: requires a stack; known hard for fixed-depth
      transformers, interesting where they fail.
    - builds on: bracket matching.

27. **Modular group composition (chain of permutations)**
    - I/O: sequence of group elements → product.
    - Why interesting: associative composition — does the model learn a
      logarithmic-depth tree or a linear scan?
    - builds on: modular addition, lookup.

28. **Binary tree path / traversal**
    - I/O: tree encoded as sequence, query node → path.
    - Why interesting: pointer-chasing through the residual stream.
    - builds on: lookup, previous-token.

---

## Tier 5 — Dynamic programming over sequences

Two-sequence problems whose textbook solution is an O(nm) DP table. The
question is whether the model represents the table explicitly, or finds a
shortcut.

29. **Longest common subsequence (LCS)**
    - I/O: two strings → length / alignment.
    - builds on: equality, span attention.

30. **Edit distance (Levenshtein)**
    - I/O: two strings → minimum edit count + operations.
    - builds on: LCS, argmin.

31. **Dynamic time warping (DTW)**
    - I/O: two real-valued sequences → warping path.
    - Why interesting: continuous cousin of edit distance; natural target for
      audio-aligned probing.
    - builds on: edit distance.

32. **Needleman–Wunsch / Smith–Waterman (global / local alignment)**
    - I/O: two sequences + scoring → alignment.
    - builds on: edit distance.

33. **Viterbi decoding (HMM most-likely path)**
    - I/O: observation sequence + transition/emission tables → state path.
    - Why interesting: max-product over a lattice — does attention learn the
      lattice structure?
    - builds on: argmax, FSM simulation.

---

## Tier 6 — Graph & search algorithms

The input is now a graph (often a 2-D grid encoded as tokens). These are the
first tasks where iterative depth — number of layers — directly bounds what
the model can express.

34. **Connected components on a 2-D grid**
    - I/O: grid of free/blocked cells → component label per cell.
    - Why interesting: union–find-like; tests whether layers implement
      iterative merging.
    - builds on: equality, 2-D block attention.

35. **BFS shortest path on an unweighted maze**
    - I/O: grid + (start, goal) → distance / path.
    - Why interesting: depth-of-network ↔ depth-of-search trade-off.
    - builds on: connected components, distance comparison.

36. **Dijkstra on a weighted 2-D maze**
    - I/O: grid with edge weights + (start, goal) → shortest path.
    - Why interesting: requires a priority-queue-like argmin over a frontier.
    - builds on: BFS, argmin.

37. **A\* with a learned / given heuristic**
    - I/O: grid + heuristic + (start, goal) → path.
    - Why interesting: tests whether the model can combine a value head with
      a search loop.
    - builds on: Dijkstra.

38. **Topological sort of a DAG**
    - I/O: edge list → linear order.
    - builds on: counting (in-degree), argmin.

---

## Tier 7 — Combinatorial search & parsing

The frontier: tasks where even classical algorithms are non-trivial. Useful
as stretch goals and as natural-failure baselines.

39. **CFG parsing (CYK over a small grammar)**
    - I/O: string + grammar → parse tree.
    - Why interesting: span-based DP over a chart; the cleanest test of
      hierarchical span composition.
    - builds on: span attention, sequence DP.

40. **Boolean satisfiability over small CNF**
    - I/O: CNF formula → satisfying assignment or UNSAT.
    - builds on: AND/OR/NOT, search.

41. **Constraint propagation (Sudoku-style)**
    - I/O: partial grid + rules → solved grid.
    - builds on: 2-D block attention, SAT.

42. **0-1 Knapsack DP**
    - I/O: (weights, values, capacity) → optimum.
    - builds on: edit-distance-style DP, argmax.

43. **Recursive descent over a CFG (generate, not parse)**
    - I/O: grammar + seed → string.
    - Why interesting: closes the loop with Tier 4 — stateful generation
      driven by a learned stack.
    - builds on: CFG parsing, Dyck.

---

## How to use this list

- For mechanistic interp: pick a block, train the smallest model that solves
  it cleanly, and write the circuit down. Lower tiers are tractable today;
  Tier 5+ are open problems.
- For curriculum / scaling experiments: a model that solves Tier *n* but
  fails at Tier *n+1* localises the missing capability.
- When proposing a new attempt under `experiments/`, cite the block it
  targets so cross-attempt comparison is possible.
