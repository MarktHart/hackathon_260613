# Problems for Attention

A roughly-sorted-by-difficulty catalogue of computational problems for a
**basic transformer architecture** — standard self-attention and a MLP per
block, and a residual stream. The goal of this list is
*not* to prescribe how to solve any of these problems. We want agents to
discover how to build them, primarily out of attention.

Conventions:
- Each task has a unique folder slug (e.g. `attention_and`). Experiments
  under `experiments/` are named with these slugs so each attempt maps
  back to exactly one problem on this list.
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

1. **AND over two tokens** — `attention_and`
   - I/O: two indicator features → 1 iff both are present.
   - What makes it hard: combining *two distinct content cues* into a
     single decision in one head, sharply enough to fire only when
     *both* fire and not when only one does.

2. **OR over two tokens** — `attention_or`
   - I/O: two indicator features → 1 iff either is present.
   - What makes it hard: the easier sibling of AND, but still requires
     composing two content cues into a single decision; the failure mode
     is collapsing to "always on" rather than "either".

3. **NOT / negation** — `attention_not`
   - I/O: one indicator → its negation.
   - What makes it hard: nothing — negation is linear. Included as a
     sanity check that the agent finds the trivial solution rather than
     overbuilding.

4. **XOR / parity over two tokens** — `attention_xor`
   - I/O: two bits → one bit.
   - What makes it hard: not linearly separable, so a single linear
     score cannot rank the three input cases in the order the answer
     requires.

5. **Equality check (token == token)** — `attention_equality`
   - I/O: (x, y) → 1[x = y].
   - What makes it hard: requires a *sharp* same/different signal, not
     a graded similarity score — and is a precondition for almost
     everything above.

6. **Linear sum over a window** — `attention_linear_sum`
   - I/O: vector of k scalars → sum.
   - What makes it hard: the natural averaging behaviour of softmax has
     to produce a true sum, not a mean.

7. **Dot product / weighted sum** — `attention_dot_product`
   - I/O: two vectors of k scalars → scalar.
   - What makes it hard: pairs each pair of positions multiplicatively,
     a different shape of computation from a plain sum.
   - builds on: linear sum.

8. **Modular addition (a + b) mod p** — `attention_modular_add`
   - I/O: (a, b) → (a+b) mod p.
   - What makes it hard: routing is trivial; the hard part is what to
     compute on the gathered operands — closed-form interpretable
     structure, classic grokking target.

9. **One-hot lookup / dispatch** — `attention_one_hot`
   - I/O: (key, table) → table[key].
   - What makes it hard: needs key-conditioned routing sharp enough to
     isolate one row without leaking from neighbours.

10. **Sign / threshold on a scalar** — `attention_sign_threshold`
    - I/O: scalar x → 1[x > θ].
    - What makes it hard: turning a soft score into a hard classification
      at a learned threshold.

---

## Tier 1 — Single-axis positional problems

Problems whose only difficulty is "where to look". The content function
is trivial; the positional / relative-distance reasoning is the whole
point.

11. **Identity copy (token at i → output at i)** — `attention_identity_copy`
    - I/O: sequence → same sequence.
    - What makes it hard: nothing — baseline that confirms the value
      path carries information unmodified.

12. **Shift-by-k (output[i] = input[i − k])** — `attention_shift_by_k`
    - I/O: sequence, fixed k → shifted sequence.
    - What makes it hard: requires a learned *relative position* signal
      precise enough to pick exactly the right offset.
    - builds on: equality.

13. **Previous-token routing** — `attention_previous_token`
    - I/O: at each position, retrieve the previous token.
    - What makes it hard: simplest non-identity positional task; the
      stepping stone for most multi-step circuits.

14. **Distance / magnitude comparison (|i − j| < r)** — `attention_distance_compare`
    - I/O: pair of positions → 1 if within radius r.
    - What makes it hard: combines positional reasoning with a
      threshold; "near *and* matching" forces composition with content
      cues.
    - builds on: AND, shift-by-k.

15. **Argmax over a window** — `attention_argmax`
    - I/O: sequence of scalars → index of maximum.
    - What makes it hard: needs a peaked attention pattern that survives
      ties and noise.

16. **Argmin over a window** — `attention_argmin`
    - I/O: sequence of scalars → index of minimum.
    - What makes it hard: symmetric to argmax under sign flip, but worth
      isolating because downstream problems (sorting, Dijkstra, MST)
      name it explicitly and it is a small, real test that the model
      can route on the *other* end of a scale.

17. **Counting (how many times does X occur in context)** — `attention_count`
    - I/O: sequence, query token → count.
    - What makes it hard: counts need *sums* across many keys, but
      attention's normaliser fights summation. May or may not be cleanly
      solvable by attention at fixed length — open for the agent to
      figure out.

18. **Running / prefix sum** — `attention_prefix_sum`
    - I/O: sequence of scalars → cumulative sum.
    - What makes it hard: every position needs an answer that depends on
      all earlier positions, with the answer changing monotonically
      along the sequence.
    - builds on: linear sum.

19. **Median / quantile over a window** — `attention_quantile`
    - I/O: window of scalars → median (or k-th percentile).
    - What makes it hard: order statistics — not expressible as a simple
      weighted average.
    - builds on: argmax, argmin.

20. **Mode / most common token in a window** — `attention_mode`
    - I/O: window of tokens → most frequent.
    - What makes it hard: requires aggregating *counts* and then
      selecting the max — two awkward operations stacked.
    - builds on: counting, argmax.

---

## Tier 2 — Span & region problems

Problems that pick out a contiguous block. The natural next step once
you can both *select content* and *reason about position*.

21. **Boundary detection (start-of-span, end-of-span markers)** — `attention_boundary`
    - I/O: sequence with delimiters → indicator at each boundary.
    - What makes it hard: needs a sharp content-keyed signal localised
      to specific positions.
    - builds on: equality, previous-token.

22. **Attend to (start, end) span** — `attention_span`
    - I/O: (start_idx, end_idx, sequence) → pooled span content.
    - What makes it hard: every position has to know whether it is
      inside a dynamically-specified range — a per-token AND of two
      positional conditions.
    - builds on: distance comparison, boundary detection.

23. **Range query / windowed sum over (a, b)** — `attention_range_sum`
    - I/O: (a, b, sequence) → sum of values in [a, b].
    - What makes it hard: combines span attention with a sum (Tier 0/1
      hardness) — masking and summing under one head.
    - builds on: attend-to-span, running sum.

24. **Bracket / parenthesis matching (depth tracking)** — `attention_brackets`
    - I/O: sequence of `( )` → depth at each position, or matching
      index.
    - What makes it hard: depth is a running counter, not a content
      lookup — pure positional or pure content attention struggles.
    - builds on: boundary detection, counting.

25. **Palindrome / symmetry detection** — `attention_palindrome`
    - I/O: sequence → 1 if it is a palindrome.
    - What makes it hard: requires pairing position i with position
      (n − i) — a learned reflection — and ANDing equalities across the
      sequence.
    - builds on: equality, distance comparison.

26. **2-D (a, b, c, d) block attention (image / spectrogram patches)** — `attention_block_2d`
    - I/O: (top, bottom, left, right, grid) → pooled block.
    - What makes it hard: factorises into two 1-D span attentions but
      needs both axes to compose cleanly.
    - builds on: span attention.

27. **Hierarchical pooling (window → segment → sequence)** — `attention_hierarchical_pool`
    - I/O: sequence → multi-scale summary.
    - What makes it hard: multiple granularities have to coexist in the
      residual stream without overwriting each other.
    - builds on: span attention.

---

## Tier 3 — Multi-token composed problems

Problems that compose Tier-1/2 primitives into the workhorse patterns we
already see in real LMs.

28. **Induction head (A B … A → B)** — `attention_induction`
    - I/O: sequence with a repeated bigram → predict the second
      occurrence's continuation.
    - What makes it hard: needs a two-step routing — "find the previous
      occurrence of the current token, then look at what followed it".
    - builds on: previous-token, equality.

29. **N-gram match with skip / wildcard** — `attention_wildcard_ngram`
    - I/O: pattern with one wildcard → matches in sequence.
    - What makes it hard: routing has to be selectively *insensitive* at
      one position while still sharp at the others.
    - builds on: induction.

30. **Substring search (multi-token pattern)** — `attention_substring`
    - I/O: (pattern, sequence) → indices of matches.
    - What makes it hard: a multi-position conjunction over a
      learned-length window; no single-head pattern is obviously enough.
    - builds on: induction, equality.

31. **Sequence reversal** — `attention_reverse`
    - I/O: x1…xn → xn…x1.
    - What makes it hard: position i must attend to position (n − i),
      which is awkward under causal masking and changes with sequence
      length.
    - builds on: distance comparison.

32. **Set deduplication / unique tokens** — `attention_dedupe`
    - I/O: sequence → first occurrences only.
    - What makes it hard: each position needs a "have I seen this
      before?" decision aggregated over all earlier positions.
    - builds on: equality, OR.

33. **Sorting (small k)** — `attention_sort`
    - I/O: k scalars → sorted k scalars.
    - What makes it hard: requires per-rank specialisation; each output
      position has a *different* selection criterion.
    - builds on: argmin, counting.

34. **Selection (k-th smallest)** — `attention_kth_select`
    - I/O: sequence of scalars, k → k-th smallest value.
    - What makes it hard: k changes the selection criterion at runtime;
      the model has to parametrise its attention on k.
    - builds on: sorting, counting.

35. **Histogram / bag-of-counts** — `attention_histogram`
    - I/O: sequence → vector of counts per vocab item.
    - What makes it hard: permutation-invariant readout with a separate
      count per bin — runs into the same averaging-vs-summing tension as
      Tier 1 counting.
    - builds on: counting.

36. **Anagram detection** — `attention_anagram`
    - I/O: two strings → 1 if they are anagrams.
    - What makes it hard: requires comparing two histograms; two layers
      of aggregation chained.
    - builds on: histogram, equality.

37. **Longest run / streak** — `attention_longest_run`
    - I/O: sequence → length of longest consecutive same-token run.
    - What makes it hard: needs to track a running counter that resets
      on a content change, then take a max.
    - builds on: bracket matching, argmax.

---

## Tier 4 — Stateful / recursive problems

The model now needs to carry running state across positions. This is
where the "obvious" attention patterns start to strain and the agent has
to discover something more interesting.

38. **Integer addition with carry** — `attention_int_add`
    - I/O: two digit sequences → their sum.
    - What makes it hard: carries propagate variable distances; the
      model has to express a learned recurrence that generalises with
      length.
    - builds on: modular addition, previous-token.

39. **Integer multiplication** — `attention_int_mul`
    - I/O: two digit sequences → product.
    - What makes it hard: nested addition with carries; each output
      digit depends on many input pairs.
    - builds on: integer addition.

40. **Finite-state machine simulation (regular language acceptance)** — `attention_fsm`
    - I/O: string, FSM spec → accept/reject.
    - What makes it hard: a hidden state has to be carried across the
      sequence, with transitions keyed on the current token.
    - builds on: lookup, equality.

41. **Regular expression matching** — `attention_regex`
    - I/O: (regex, string) → match indicator.
    - What makes it hard: a per-input FSM the model has to build at
      inference time, not a fixed one.
    - builds on: FSM simulation.

42. **Dyck-k language (nested brackets of k types)** — `attention_dyck`
    - I/O: string of brackets → well-formed?.
    - What makes it hard: depth tracking with type-matching at every
      close bracket — classically hard at fixed depth.
    - builds on: bracket matching.

43. **Modular group composition (chain of permutations)** — `attention_group_compose`
    - I/O: sequence of group elements → product.
    - What makes it hard: associative composition over many positions;
      sequential scan vs parallel tree is up to the model to find.
    - builds on: modular addition, lookup.

44. **Binary tree path / traversal** — `attention_tree_path`
    - I/O: tree encoded as sequence, query node → path.
    - What makes it hard: pointer-chasing of variable depth — each step
      depends on the result of the previous one.
    - builds on: lookup, previous-token.

45. **GCD / Euclidean algorithm** — `attention_gcd`
    - I/O: (a, b) → gcd(a, b).
    - What makes it hard: variable-length iterative reduction whose
      depth depends on the inputs.
    - builds on: integer addition.

46. **Game of Life — one step** — `attention_game_of_life`
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

47. **Longest common subsequence (LCS)** — `attention_lcs`
    - I/O: two strings → length / alignment.
    - builds on: equality, span attention.

48. **Edit distance (Levenshtein)** — `attention_edit_distance`
    - I/O: two strings → minimum edit count + operations.
    - builds on: LCS, argmin.

49. **Dynamic time warping (DTW)** — `attention_dtw`
    - I/O: two real-valued sequences → warping path.
    - builds on: edit distance.

50. **Needleman–Wunsch (global alignment)** — `attention_global_align`
    - I/O: two sequences + scoring → end-to-end alignment.
    - What makes it hard: a DP that has to commit to aligning *both*
      sequences in their entirety — every cell's optimum depends on the
      full upstream table.
    - builds on: edit distance.

51. **Smith–Waterman (local alignment)** — `attention_local_align`
    - I/O: two sequences + scoring → best-scoring local alignment.
    - What makes it hard: a DP with a learned 0-floor — the model has
      to discover when to *restart* the alignment, not just extend it.
    - builds on: edit distance.

52. **Viterbi decoding (HMM most-likely path)** — `attention_viterbi`
    - I/O: observation sequence + transition/emission tables → state
      path.
    - builds on: argmax, FSM simulation.

53. **Longest increasing subsequence** — `attention_lis`
    - I/O: sequence of scalars → length / indices.
    - builds on: counting, argmax.

54. **Optimal binary search tree** — `attention_optimal_bst`
    - I/O: keys + access weights → optimal BST split structure.
    - What makes it hard: a span DP whose split point is *chosen*, not
      derived from the data — every interval optimises over its own
      partitions.
    - builds on: span attention, argmin.

55. **Matrix chain order** — `attention_matrix_chain`
    - I/O: matrix dimensions → optimal parenthesisation.
    - What makes it hard: same span-DP shape as optimal BST but with a
      multiplicative cost; included because the agent may discover
      the shared structure (or not).
    - builds on: span attention, argmin.

---

## Tier 6 — Graph & search problems

The input is now a graph (often a 2-D grid encoded as tokens). These are
the first problems where iterative depth — the number of transformer
layers — directly bounds what the model can express.

56. **Connected components on a 2-D grid** — `attention_connected_components`
    - I/O: grid of free/blocked cells → component label per cell.
    - builds on: equality, 2-D block attention.

57. **BFS shortest path on an unweighted maze** — `attention_bfs`
    - I/O: grid + (start, goal) → distance / path.
    - builds on: connected components, distance comparison.

58. **Dijkstra on a weighted 2-D maze** — `attention_dijkstra`
    - I/O: grid with edge weights + (start, goal) → shortest path.
    - builds on: BFS, argmin.

59. **A\* with a learned / given heuristic** — `attention_astar`
    - I/O: grid + heuristic + (start, goal) → path.
    - builds on: Dijkstra.

60. **Topological sort of a DAG** — `attention_topo_sort`
    - I/O: edge list → linear order.
    - builds on: counting (in-degree), argmin.

61. **Bipartite matching** — `attention_bipartite`
    - I/O: bipartite edge list → maximum matching.
    - builds on: topological sort, search.

62. **Minimum spanning tree** — `attention_mst`
    - I/O: weighted edge list → MST edges.
    - builds on: Dijkstra, sorting.

63. **Strongly connected components** — `attention_scc`
    - I/O: directed edge list → SCC labels.
    - builds on: connected components, traversal.

---

## Tier 7 — Combinatorial search & open problems

The frontier: problems whose classical solutions are non-trivial, and
where it is genuinely unclear whether a small basic transformer can
express them. Useful as stretch goals and as natural-failure baselines.

64. **CFG parsing (CYK over a small grammar)** — `attention_cyk`
    - I/O: string + grammar → parse tree.
    - builds on: span attention, sequence DP.

65. **Recursive descent over a CFG (generate, not parse)** — `attention_cfg_generate`
    - I/O: grammar + seed → string.
    - builds on: CFG parsing, Dyck.

66. **Boolean satisfiability over small CNF** — `attention_sat`
    - I/O: CNF formula → satisfying assignment or UNSAT.
    - builds on: AND, OR, NOT, search.

67. **Constraint propagation (Sudoku-style)** — `attention_constraint_prop`
    - I/O: partial grid + rules → solved grid.
    - builds on: 2-D block attention, SAT.

68. **0-1 Knapsack DP** — `attention_knapsack`
    - I/O: (weights, values, capacity) → optimum.
    - builds on: edit-distance-style DP, argmax.

69. **Graph colouring (small chromatic number)** — `attention_graph_color`
    - I/O: graph → valid k-colouring or fail.
    - builds on: SAT.

70. **Travelling salesman (small n)** — `attention_tsp`
    - I/O: distance matrix → optimal tour.
    - Open whether a basic transformer expresses optimal search or
      learns only heuristic tours.
    - builds on: Dijkstra, search.

71. **Polynomial evaluation (Horner's method)** — `attention_polyeval`
    - I/O: coefficients + x → value.
    - builds on: integer multiplication, running sum.

72. **Small-matrix multiplication** — `attention_matmul`
    - I/O: two small matrices → product.
    - builds on: dot product.

73. **Game tree minimax (small depth)** — `attention_minimax`
    - I/O: small game state → optimal move.
    - builds on: tree traversal, argmax, argmin.

---

## How to use this list

- For mechanistic interp: pick a problem and train the smallest basic
  transformer that solves it, then write down which part of the
  architecture is doing the work. The interesting outputs are not just
  "did it solve it" but *how*.
- For curriculum / scaling experiments: a basic transformer that
  solves Tier *n* but struggles at Tier *n+1* localises the harder
  capability.
- When proposing a new attempt under `experiments/`, name the folder
  with the task's slug (e.g. `experiments/attention_induction/`) so
  cross-attempt comparison is possible.
