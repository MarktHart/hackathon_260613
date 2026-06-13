import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Tuple

# Fixed Dyck-1 vocabulary (single bracket type). See README "Canonical
# measurement condition".
VOCAB = {"pad": 0, "open": 1, "close": 2}
PAD = VOCAB["pad"]
OPEN = VOCAB["open"]
CLOSE = VOCAB["close"]

SEQ_LEN = 32          # all sequences padded/built to this length
N_SEQ = 256           # batch size
N_PAIRS = SEQ_LEN // 2  # 16 bracket pairs -> exactly fills SEQ_LEN, no PAD
CANONICAL_DEPTH = 3   # depth used for the headline metric
SWEEP_DEPTHS = [1, 2, 3, 4, 5]

# model_fn returns a dict with a single "attention" tensor.
ModelFn = Callable[[np.ndarray], Dict[str, np.ndarray]]


@dataclass(frozen=True)
class Batch:
    tokens: np.ndarray              # (batch, seq_len) int32, values in {0,1,2}
    # pairs[b] is a list of (open_pos, close_pos, depth) for sequence b
    pairs: List[List[Tuple[int, int, int]]]


def _gen_dyck1(rng: np.random.Generator, n_pairs: int, max_depth: int) -> List[int]:
    """Generate one valid Dyck-1 sequence with exactly `n_pairs` pairs and
    nesting capped at `max_depth`. Length is exactly 2 * n_pairs."""
    seq: List[int] = []
    stack_depth = 0
    opens_remaining = n_pairs
    closes_remaining = n_pairs

    while opens_remaining + closes_remaining > 0:
        can_open = opens_remaining > 0 and stack_depth < max_depth
        can_close = stack_depth > 0

        if can_open and can_close:
            action = rng.choice(["open", "close"])
        elif can_open:
            action = "open"
        elif can_close:
            action = "close"
        else:
            # Defensive: should not happen for max_depth >= 1, but bail safely.
            break

        if action == "open":
            seq.append(OPEN)
            stack_depth += 1
            opens_remaining -= 1
        else:
            seq.append(CLOSE)
            stack_depth -= 1
            closes_remaining -= 1

    return seq


def _pairs_and_depths(seq: List[int]) -> List[Tuple[int, int, int]]:
    """Return (open_pos, close_pos, depth) for every bracket pair.

    `depth` is the nesting depth of the matched pair: the number of brackets
    open at the moment the pair's `)` is emitted, counting itself (1-indexed).
    """
    stack: List[int] = []
    pairs: List[Tuple[int, int, int]] = []
    for i, t in enumerate(seq):
        if t == OPEN:
            stack.append(i)
        elif t == CLOSE:
            open_pos = stack.pop()
            depth = len(stack) + 1
            pairs.append((open_pos, i, depth))
    return pairs


def generate(seed: int = 42) -> Batch:
    """Deterministic CFG batch generation.

    Returns 256 valid Dyck-1 sequences of length 32. Target nesting depths are
    spread 1..5 so that every sweep depth is well sampled. Same seed -> same
    batch.
    """
    rng = np.random.default_rng(seed)

    # Deterministic spread of target max-depths across the batch, then shuffle.
    target_depths = (
        [1] * 51 + [2] * 51 + [3] * 52 + [4] * 51 + [5] * 51
    )  # sums to 256
    assert len(target_depths) == N_SEQ
    rng.shuffle(target_depths)

    tokens = np.zeros((N_SEQ, SEQ_LEN), dtype=np.int32)
    pairs: List[List[Tuple[int, int, int]]] = []

    for b, td in enumerate(target_depths):
        seq = _gen_dyck1(rng, n_pairs=N_PAIRS, max_depth=int(td))
        # seq has length exactly SEQ_LEN; assign directly (no PAD needed).
        tokens[b, : len(seq)] = seq
        pairs.append(_pairs_and_depths(seq))

    return Batch(tokens=tokens, pairs=pairs)


def evaluate(model_fn: ModelFn) -> Dict[str, Any]:
    """Run `model_fn` on the canonical batch and return the scoring payload.

    The payload shape is documented in the goal README ("Payload contract").
    """
    batch = generate(seed=42)
    out = model_fn(batch.tokens)

    # ---- validate model_fn output ----
    if not isinstance(out, dict) or "attention" not in out:
        raise ValueError(
            "model_fn must return a dict with key 'attention'; got "
            f"{type(out).__name__} with keys "
            f"{list(out.keys()) if isinstance(out, dict) else 'N/A'}"
        )
    attn = np.asarray(out["attention"], dtype=np.float64)
    if attn.ndim != 4:
        raise ValueError(
            f"attention must be 4-D [batch, n_heads, seq, seq]; got shape {attn.shape}"
        )
    n_batch, n_heads, s1, s2 = attn.shape
    if n_batch != N_SEQ or s1 != SEQ_LEN or s2 != SEQ_LEN:
        raise ValueError(
            f"attention shape {attn.shape} incompatible with batch "
            f"[{N_SEQ}, n_heads, {SEQ_LEN}, {SEQ_LEN}]"
        )

    # Number of non-PAD tokens in the causal prefix of each position.
    # PAD only ever appears as a trailing run, so the prefix [0..i] of a
    # non-PAD position i contains i+1 non-PAD tokens.
    non_pad = batch.tokens != PAD  # (batch, seq)
    prefix_non_pad = np.cumsum(non_pad, axis=1)  # (batch, seq); count incl. self

    # Accumulate per-depth measurements.
    sums_match: Dict[int, float] = {d: 0.0 for d in SWEEP_DEPTHS}
    sums_uniform: Dict[int, float] = {d: 0.0 for d in SWEEP_DEPTHS}
    counts: Dict[int, int] = {d: 0 for d in SWEEP_DEPTHS}

    for b in range(N_SEQ):
        for open_pos, close_pos, depth in batch.pairs[b]:
            if depth not in sums_match:
                continue  # depths beyond the sweep are not measured
            # Attention from the closing token to its matching opening token,
            # averaged over heads.
            attn_to_match = float(np.mean(attn[b, :, close_pos, open_pos]))
            denom = int(prefix_non_pad[b, close_pos])
            uniform = 1.0 / denom if denom > 0 else 0.0
            sums_match[depth] += attn_to_match
            sums_uniform[depth] += uniform
            counts[depth] += 1

    sweep = []
    for d in SWEEP_DEPTHS:
        n = counts[d]
        if n == 0:
            sweep.append({
                "depth": d,
                "n_pairs": 0,
                "mean_attn_to_match": 0.0,
                "mean_attn_uniform": 0.0,
            })
        else:
            sweep.append({
                "depth": d,
                "n_pairs": int(n),
                "mean_attn_to_match": float(sums_match[d] / n),
                "mean_attn_uniform": float(sums_uniform[d] / n),
            })

    return {
        "version": 1,
        "canonical_depth": CANONICAL_DEPTH,
        "seq_len": SEQ_LEN,
        "vocab": dict(VOCAB),
        "sweep": sweep,
        "model_metadata": {
            "n_heads": int(n_heads),
            "batch_size": int(n_batch),
        },
    }


# ---- reference model functions (smoke testing only; not used by score) ----

def uniform_attention_model_fn(n_heads: int = 4) -> ModelFn:
    """A strawman: causal-uniform attention (each query attends equally over its
    prefix). Its mean_attn_to_match equals the uniform baseline at every depth.
    Pure NumPy, no GPU."""
    def _fn(tokens: np.ndarray) -> Dict[str, np.ndarray]:
        b, s = tokens.shape
        attn = np.zeros((b, n_heads, s, s), dtype=np.float32)
        for i in range(s):
            attn[:, :, i, : i + 1] = 1.0 / (i + 1)
        return {"attention": attn}
    return _fn


def stack_oracle_model_fn(n_heads: int = 4) -> ModelFn:
    """An oracle: every closing token puts all attention mass on its matching
    opening token. Achieves mean_attn_to_match == 1.0 at every depth. Pure
    NumPy, no GPU."""
    def _fn(tokens: np.ndarray) -> Dict[str, np.ndarray]:
        b, s = tokens.shape
        attn = np.zeros((b, n_heads, s, s), dtype=np.float32)
        for bi in range(b):
            # default each query to self-attention so rows still sum to 1
            for i in range(s):
                attn[bi, :, i, i] = 1.0
            for open_pos, close_pos, _depth in _pairs_and_depths(list(tokens[bi])):
                attn[bi, :, close_pos, :] = 0.0
                attn[bi, :, close_pos, open_pos] = 1.0
        return {"attention": attn}
    return _fn
