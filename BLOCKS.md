# Problems for Attention

A roughly-sorted-by-difficulty catalogue of computational problems for a
**basic transformer architecture** — standard self-attention and a MLP per
block, and a residual stream. The goal of this list is
*not* to prescribe how to solve any of these problems. We want agents to
discover how to build them, primarily out of attention.

Conventions:
- **I/O** sketches the computation as (input → output).
- **What makes it hard** says — in one or two lines — why this is not
  trivial. It deliberately *does not* name a mechanism, circuit, or which
  component of the transformer should carry the work. That is the
  discovery part.
- Within a tier, items are also roughly ordered by difficulty.
- Higher tiers tend to involve lower ones — dependencies are noted in
  `builds on:` lines where they are reasonably clear.

Nothing on this list is "impossible for attention". Every entry is a hard
problem; some are likely to need cleverness beyond the obvious patterns;
some may turn out not to be solvable by the smallest basic transformer
and force the agent to motivate a small, well-defined extension. That is
fine — the discovery is the point.

---

## Tier 0 — Boolean & arithmetic primitives

The smallest unit tests. A basic transformer should solve these crisply;
anything higher up will only be as clean as Tier 0 is.

1. **AND / OR over two tokens**
   - I/O: two indicator features → one indicator.
   - What makes it hard: combining *two distinct content cues* into a
     single decision in one head, sharply enough to distinguish "both"
     from "either".

2. **NOT / negation**
   - I/O: one indicator → its negation.
   - What makes it hard: nothing — negation is linear. Included as a
     sanity check that the agent finds the trivial solution rather than
     overbuilding.

3. **XOR / parity over two tokens**
   - I/O: two bits → one bit.
   - What makes it hard: not linearly separable, so a single linear
     score cannot rank the three input cases in the order the answer
     requires.

4. **Equality check (token == token)**
   - I/O: (x, y) → 1[x = y].
   - What makes it hard: requires a *sharp* same/different signal, not
     a graded similarity score — and is a precondition for almost
     everything above.

5. **Linear sum over a window**
   - I/O: vector of k scalars → sum.
   - What makes it hard: the natural averaging behaviour of softmax has
     to produce a true sum, not a mean.

6. **Dot product / weighted sum**
   - I/O: two vectors of k scalars → scalar.
   - What makes it hard: pairs each pair of positions multiplicatively,
     a different shape of computation from a plain sum.
   - builds on: linear sum.

7. **Modular addition (a + b) mod p**
   - I/O: (a, b) → (a+b) mod p.
   - What makes it hard: routing is trivial; the hard part is what to
     compute on the gathered operands — closed-form interpretable
     structure, classic grokking target.

8. **One-hot lookup / dispatch**
   - I/O: (key, table) → table[key].
   - What makes it hard: needs key-conditioned routing sharp enough to
     isolate one row without leaking from neighbours.

9. **Sign / threshold on a scalar**
   - I/O: scalar x → 1[x > θ].
   - What makes it hard: turning a soft score into a hard classification
     at a learned threshold.

---

## Tier 1 — Single-axis positional problems

Problems whose only difficulty is "where to look". The content function
is trivial; the positional / relative-distance reasoning is the whole
point.

10. **Identity copy (token at i → output at i)**
    - I/O: sequence → same sequence.
    - What makes it hard: nothing — baseline that confirms the value
      path carries information unmodified.

11. **Shift-by-k (output[i] = input[i − k])**
    - I/O: sequence, fixed k → shifted sequence.
    - What makes it hard: requires a learned *relative position* signal
      precise enough to pick exactly the right offset.
    - builds on: equality.

12. **Previous-token routing**
    - I/O: at each position, retrieve the previous token.
    - What makes it hard: simplest non-identity positional task; the
      stepping stone for most multi-step circuits.

13. **Distance / magnitude comparison (|i − j| < r)**
    - I/O: pair of positions → 1 if within radius r.
    - What makes it hard: combines positional reasoning with a
      threshold; "near *and* matching" forces composition with content
      cues.
    - builds on: AND, shift-by-k.

14. **Argmax / argmin over a window**
    - I/O: sequence of scalars → index of extremum.
    - What makes it hard: needs a peaked attention pattern that survives
      ties and noise.

15. **Counting (how many times does X occur in context)**
    - I/O: sequence, query token → count.
    - What makes it hard: counts need *sums* across many keys, but
      attention's normaliser fights summation. May or may not be cleanly
      solvable by attention at fixed length — open for the agent to
      figure out.

16. **Running / prefix sum**
    - I/O: sequence of scalars → cumulative sum.
    - What makes it hard: every position needs an answer that depends on
      all earlier positions, with the answer changing monotonically
      along the sequence.
    - builds on: linear sum.

17. **Median / quantile over a window**
    - I/O: window of scalars → median (or k-th percentile).
    - What makes it hard: order statistics — not expressible as a simple
      weighted average.
    - builds on: argmax/argmin.

18. **Mode / most common token in a window**
    - I/O: window of tokens → most frequent.
    - What makes it hard: requires aggregating *counts* and then
      selecting the max — two awkward operations stacked.
    - builds on: counting, argmax.

---

## Tier 2 — Span & region problems

Problems that pick out a contiguous block. The natural next step once
you can both *select content* and *reason about position*.

19. **Boundary detection (start-of-span, end-of-span markers)**
    - I/O: sequence with delimiters → indicator at each boundary.
    - What makes it hard: needs a sharp content-keyed signal localised
      to specific positions.
    - builds on: equality, previous-token.

20. **Attend to (start, end) span**
    - I/O: (start_idx, end_idx, sequence) → pooled span content.
    - What makes it hard: every position has to know whether it is
      inside a dynamically-specified range — a per-token AND of two
      positional conditions.
    - builds on: distance comparison, boundary detection.

21. **Range query / windowed sum over (a, b)**
    - I/O: (a, b, sequence) → sum of values in [a, b].
    - What makes it hard: combines span attention with a sum (Tier 0/1
      hardness) — masking and summing under one head.
    - builds on: attend-to-span, running sum.

22. **Bracket / parenthesis matching (depth tracking)**
    - I/O: sequence of `( )` → depth at each position, or matching
      index.
    - What makes it hard: depth is a running counter, not a content
      lookup — pure positional or pure content attention struggles.
    - builds on: boundary detection, counting.

23. **Palindrome / symmetry detection**
    - I/O: sequence → 1 if it is a palindrome.
    - What makes it hard: requires pairing position i with position
      (n − i) — a learned reflection — and ANDing equalities across the
      sequence.
    - builds on: equality, distance comparison.

24. **2-D (a, b, c, d) block attention (image / spectrogram patches)**
    - I/O: (top, bottom, left, right, grid) → pooled block.
    - What makes it hard: factorises into two 1-D span attentions but
      needs both axes to compose cleanly.
    - builds on: span attention.

25. **Hierarchical pooling (window → segment → sequence)**
    - I/O: sequence → multi-scale summary.
    - What makes it hard: multiple granularities have to coexist in the
      residual stream without overwriting each other.
    - builds on: span attention.

---

## Tier 3 — Multi-token composed problems

Problems that compose Tier-1/2 primitives into the workhorse patterns we
already see in real LMs.

26. **Induction head (A B … A → B)**
    - I/O: sequence with a repeated bigram → predict the second
      occurrence's continuation.
    - What makes it hard: needs a two-step routing — "find the previous
      occurrence of the current token, then look at what followed it".
    - builds on: previous-token, equality.

27. **N-gram match with skip / wildcard**
    - I/O: pattern with one wildcard → matches in sequence.
    - What makes it hard: routing has to be selectively *insensitive* at
      one position while still sharp at the others.
    - builds on: induction.

28. **Substring search (multi-token pattern)**
    - I/O: (pattern, sequence) → indices of matches.
    - What makes it hard: a multi-position conjunction over a
      learned-length window; no single-head pattern is obviously enough.
    - builds on: induction, equality.

29. **Sequence reversal**
    - I/O: x1…xn → xn…x1.
    - What makes it hard: position i must attend to position (n − i),
      which is awkward under causal masking and changes with sequence
      length.
    - builds on: distance comparison.

30. **Set deduplication / unique tokens**
    - I/O: sequence → first occurrences only.
    - What makes it hard: each position needs a "have I seen this
      before?" decision aggregated over all earlier positions.
    - builds on: equality, OR.

31. **Sorting (small k)**
    - I/O: k scalars → sorted k scalars.
    - What makes it hard: requires per-rank specialisation; each output
      position has a *different* selection criterion.
    - builds on: argmin, counting.

32. **Selection (k-th smallest)**
    - I/O: sequence of scalars, k → k-th smallest value.
    - What makes it hard: k changes the selection criterion at runtime;
      the model has to parametrise its attention on k.
    - builds on: sorting, counting.

33. **Histogram / bag-of-counts**
    - I/O: sequence → vector of counts per vocab item.
    - What makes it hard: permutation-invariant readout with a separate
      count per bin — runs into the same averaging-vs-summing tension as
      Tier 1 counting.
    - builds on: counting.

34. **Anagram detection**
    - I/O: two strings → 1 if they are anagrams.
    - What makes it hard: requires comparing two histograms; two layers
      of aggregation chained.
    - builds on: histogram, equality.

35. **Longest run / streak**
    - I/O: sequence → length of longest consecutive same-token run.
    - What makes it hard: needs to track a running counter that resets
      on a content change, then take a max.
    - builds on: bracket matching, argmax.

---

## Tier 4 — Stateful / recursive problems

The model now needs to carry running state across positions. This is
where the "obvious" attention patterns start to strain and the agent has
to discover something more interesting.

36. **Integer addition with carry**
    - I/O: two digit sequences → their sum.
    - What makes it hard: carries propagate variable distances; the
      model has to express a learned recurrence that generalises with
      length.
    - builds on: modular addition, previous-token.

37. **Integer multiplication**
    - I/O: two digit sequences → product.
    - What makes it hard: nested addition with carries; each output
      digit depends on many input pairs.
    - builds on: integer addition.

38. **Finite-state machine simulation (regular language acceptance)**
    - I/O: string, FSM spec → accept/reject.
    - What makes it hard: a hidden state has to be carried across the
      sequence, with transitions keyed on the current token.
    - builds on: lookup, equality.

39. **Regular expression matching**
    - I/O: (regex, string) → match indicator.
    - What makes it hard: a per-input FSM the model has to build at
      inference time, not a fixed one.
    - builds on: FSM simulation.

40. **Dyck-k language (nested brackets of k types)**
    - I/O: string of brackets → well-formed?.
    - What makes it hard: depth tracking with type-matching at every
      close bracket — classically hard at fixed depth.
    - builds on: bracket matching.

41. **Modular group composition (chain of permutations)**
    - I/O: sequence of group elements → product.
    - What makes it hard: associative composition over many positions;
      sequential scan vs parallel tree is up to the model to find.
    - builds on: modular addition, lookup.

42. **Binary tree path / traversal**
    - I/O: tree encoded as sequence, query node → path.
    - What makes it hard: pointer-chasing of variable depth — each step
      depends on the result of the previous one.
    - builds on: lookup, previous-token.

43. **GCD / Euclidean algorithm**
    - I/O: (a, b) → gcd(a, b).
    - What makes it hard: variable-length iterative reduction whose
      depth depends on the inputs.
    - builds on: integer addition.

44. **Game of Life — one step**
    - I/O: 2-D grid → next-state grid.
    - What makes it hard: every cell is a small per-neighbourhood
      function over 8 neighbours; trivial in isolation, interesting at
      scale.
    - builds on: 2-D block attention, counting.

---

## Tier 5 — Dynamic programming over sequences

Two-sequence problems whose textbook solution is an O(nm) DP table. May
or may not be solvable by a small basic transformer in the obvious way —
agent's choice whether to embed the table or find a shortcut.

45. **Longest common subsequence (LCS)**
    - I/O: two strings → length / alignment.
    - builds on: equality, span attention.

46. **Edit distance (Levenshtein)**
    - I/O: two strings → minimum edit count + operations.
    - builds on: LCS, argmin.

47. **Dynamic time warping (DTW)**
    - I/O: two real-valued sequences → warping path.
    - builds on: edit distance.

48. **Needleman–Wunsch / Smith–Waterman (global / local alignment)**
    - I/O: two sequences + scoring → alignment.
    - builds on: edit distance.

49. **Viterbi decoding (HMM most-likely path)**
    - I/O: observation sequence + transition/emission tables → state
      path.
    - builds on: argmax, FSM simulation.

50. **Longest increasing subsequence**
    - I/O: sequence of scalars → length / indices.
    - builds on: counting, argmax.

51. **Optimal binary search tree / matrix chain order**
    - I/O: weights → optimal split structure.
    - builds on: span DP.

---

## Tier 6 — Graph & search problems

The input is now a graph (often a 2-D grid encoded as tokens). These are
the first problems where iterative depth — the number of transformer
layers — directly bounds what the model can express.

52. **Connected components on a 2-D grid**
    - I/O: grid of free/blocked cells → component label per cell.
    - builds on: equality, 2-D block attention.

53. **BFS shortest path on an unweighted maze**
    - I/O: grid + (start, goal) → distance / path.
    - builds on: connected components, distance comparison.

54. **Dijkstra on a weighted 2-D maze**
    - I/O: grid with edge weights + (start, goal) → shortest path.
    - builds on: BFS, argmin.

55. **A\* with a learned / given heuristic**
    - I/O: grid + heuristic + (start, goal) → path.
    - builds on: Dijkstra.

56. **Topological sort of a DAG**
    - I/O: edge list → linear order.
    - builds on: counting (in-degree), argmin.

57. **Bipartite matching**
    - I/O: bipartite edge list → maximum matching.
    - builds on: topological sort, search.

58. **Minimum spanning tree**
    - I/O: weighted edge list → MST edges.
    - builds on: Dijkstra, sorting.

59. **Strongly connected components**
    - I/O: directed edge list → SCC labels.
    - builds on: connected components, traversal.

---

## Tier 7 — Combinatorial search & open problems

The frontier: problems whose classical solutions are non-trivial, and
where it is genuinely unclear whether a small basic transformer can
express them. Useful as stretch goals and as natural-failure baselines.

60. **CFG parsing (CYK over a small grammar)**
    - I/O: string + grammar → parse tree.
    - builds on: span attention, sequence DP.

61. **Recursive descent over a CFG (generate, not parse)**
    - I/O: grammar + seed → string.
    - builds on: CFG parsing, Dyck.

62. **Boolean satisfiability over small CNF**
    - I/O: CNF formula → satisfying assignment or UNSAT.
    - builds on: AND/OR/NOT, search.

63. **Constraint propagation (Sudoku-style)**
    - I/O: partial grid + rules → solved grid.
    - builds on: 2-D block attention, SAT.

64. **0-1 Knapsack DP**
    - I/O: (weights, values, capacity) → optimum.
    - builds on: edit-distance-style DP, argmax.

65. **Graph colouring (small chromatic number)**
    - I/O: graph → valid k-colouring or fail.
    - builds on: SAT.

66. **Travelling salesman (small n)**
    - I/O: distance matrix → optimal tour.
    - Open whether a basic transformer expresses optimal search or
      learns only heuristic tours.
    - builds on: Dijkstra, search.

67. **Polynomial evaluation (Horner's method)**
    - I/O: coefficients + x → value.
    - builds on: integer multiplication, running sum.

68. **Small-matrix multiplication**
    - I/O: two small matrices → product.
    - builds on: dot product.

69. **Game tree minimax (small depth)**
    - I/O: small game state → optimal move.
    - builds on: tree traversal, argmax/argmin.

---

## How to use this list

- For mechanistic interp: pick a problem and train the smallest basic
  transformer that solves it, then write down which part of the
  architecture is doing the work. The interesting outputs are not just
  "did it solve it" but *how*.
- For curriculum / scaling experiments: a basic transformer that
  solves Tier *n* but struggles at Tier *n+1* localises the harder
  capability.
- When proposing a new attempt under `experiments/`, cite the problem
  it targets so cross-attempt comparison is possible.
