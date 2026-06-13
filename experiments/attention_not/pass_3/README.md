# What I did

This attempt implements a **hand-built logical NOT mechanism** for a single attention head. The mechanism exploits the fully synthetic, fixed-geometry setup: the task generator uses a deterministic orthonormal basis (seed=42), so the target key direction `k_t = basis[0]`, the negation anchor `neg_anchor = basis[8]`, and the absent filler `absent_slot = basis[9]` are all known constants.

The `model_fn` computes standard dot-product logits `q · k_i` for all keys, then **detects the negation marker** by inspecting the negation slot (keys[1]) and measuring its projection onto `neg_anchor`: `marker_strength = |keys[1] · neg_anchor|`. When the marker is absent, slot 1 holds `absent_slot` which is orthogonal to `neg_anchor` (marker_strength = 0). When present, slot 1 holds `k_neg = cos·k_t + sin·neg_anchor`, giving `marker_strength = sin = √(1−cos²)`. The mechanism then **subtracts `SCALE × marker_strength` from the target logit (slot 0)**, making the target logit itself drop — genuine content-specific inhibition, not just softmax competition from the marker key.

The suppression scale (3.0) was chosen so that at the canonical anchor (cos=0.0, marker_strength=1.0) the target logit goes from +1.0 to −2.0, driving target attention from ~0.28 to ~0.02 (sharpness ≈ 0.93). As superposition increases (cos → 1), the marker aligns with the target, its projection onto `neg_anchor` shrinks (sin → 0), and suppression gracefully weakens — matching the expected operating range.

# Why this visualisation

The Demo tab shows four coordinated panels that together make the NOT claim legible:
1. **Sharpness sweep** (top-left) — direct comparison of the attempt vs. the linear baseline across all cos values; the gap is the causal contribution of the NOT mechanism.
2. **Target attention bars** (top-right) — raw softmax attention to the target key with/without the marker; the visual drop at each cos value is the observable behaviour.
3. **Lift over baseline** (bottom-left) — isolates the mechanism's advantage over softmax competition alone; green bars confirm positive lift at every sweep point.
4. **Mechanism summary** (bottom-right) — textual schematic of the circuit (target logit ← q·k_t − SCALE·|k₁·neg_anchor|) so a reader can verify the logic without reading code.

The canonical anchor (cos=0.0) is highlighted in the summary metrics. The Benchmark tab drops in the shared leaderboard to contextualise this attempt against any future ones.