import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
import hashlib

@dataclass(frozen=True)
class Batch:
    input_ids: np.ndarray          # (batch, seq_len) int32
    attention_mask: np.ndarray     # (batch, seq_len) bool
    labels: np.ndarray             # (batch, seq_len) int32: -100 for non-closing, else stack depth at that closing
    matching_open_pos: np.ndarray  # (batch, seq_len) int32: -1 for non-closing, else position of matching '('
    open_depth: np.ndarray         # (batch, seq_len) int32: -100 for non-open, else nesting depth of that '(' (1-indexed)
    max_depth: int
    seq_len: int

VOCAB = {"PAD": 0, "OPEN": 1, "CLOSE": 2, "BOS": 3, "EOS": 4}
MAX_DEPTH = 6
MAX_LEN = 64
CANONICAL_SEED = 42
TEST_SPLIT_SIZE = 512

def _generate_dyck_string(rng: np.random.Generator, max_depth: int, max_len: int) -> List[int]:
    """Generate a single valid Dyck-1 string with BOS/EOS."""
    depth = 0
    tokens = [VOCAB["BOS"]]
    while len(tokens) < max_len - 1:  # leave room for EOS
        # Can open if depth < max_depth and space remains for closing
        can_open = depth < max_depth and (len(tokens) + 1 + depth) < max_len
        # Can close if depth > 0
        can_close = depth > 0
        if not can_open and not can_close:
            break
        if can_open and can_close:
            # Bias toward closing as depth grows to avoid runaway opens
            p_open = 0.5 * (1 - depth / max_depth)
            if rng.random() < p_open:
                tokens.append(VOCAB["OPEN"])
                depth += 1
            else:
                tokens.append(VOCAB["CLOSE"])
                depth -= 1
        elif can_open:
            tokens.append(VOCAB["OPEN"])
            depth += 1
        else:
            tokens.append(VOCAB["CLOSE"])
            depth -= 1
    # Close any remaining opens
    while depth > 0 and len(tokens) < max_len - 1:
        tokens.append(VOCAB["CLOSE"])
        depth -= 1
    tokens.append(VOCAB["EOS"])
    return tokens

def _compute_labels_and_matches(tokens: List[int], max_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (labels, matching_open_pos, open_depth) padded to max_len."""
    labels = np.full(max_len, -100, dtype=np.int32)
    matching = np.full(max_len, -1, dtype=np.int32)
    open_depth = np.full(max_len, -100, dtype=np.int32)
    stack = []
    for i, tok in enumerate(tokens):
        if tok == VOCAB["OPEN"]:
            stack.append(i)
            open_depth[i] = len(stack)  # nesting depth at this open (1-indexed)
        elif tok == VOCAB["CLOSE"]:
            if stack:
                open_pos = stack.pop()
                labels[i] = len(stack) + 1  # depth after this close = previous depth
                matching[i] = open_pos
    return labels, matching, open_depth

def generate(seed: int = 0) -> Batch:
    """Deterministic Dyck batch for a given seed."""
    rng = np.random.default_rng(seed)
    sequences = []
    all_labels = []
    all_matching = []
    all_open_depth = []
    for _ in range(TEST_SPLIT_SIZE):
        toks = _generate_dyck_string(rng, MAX_DEPTH, MAX_LEN)
        seq_len = len(toks)
        # Pad to MAX_LEN
        padded = toks + [VOCAB["PAD"]] * (MAX_LEN - seq_len)
        sequences.append(padded)
        labels, matching, open_depth = _compute_labels_and_matches(toks, MAX_LEN)
        all_labels.append(labels)
        all_matching.append(matching)
        all_open_depth.append(open_depth)
    input_ids = np.array(sequences, dtype=np.int32)
    attention_mask = (input_ids != VOCAB["PAD"])
    labels = np.array(all_labels, dtype=np.int32)
    matching_open_pos = np.array(all_matching, dtype=np.int32)
    open_depth = np.array(all_open_depth, dtype=np.int32)
    return Batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        matching_open_pos=matching_open_pos,
        open_depth=open_depth,
        max_depth=MAX_DEPTH,
        seq_len=MAX_LEN,
    )

def _compute_head_metrics(attn: np.ndarray, batch: Batch) -> Dict[str, float]:
    """attn: (batch, seq_len, seq_len) for a single head."""
    batch_size, seq_len, _ = attn.shape
    matching_accs = []
    depth_corrs = []
    diag_fracs = []
    for b in range(batch_size):
        mask = batch.attention_mask[b]
        valid = np.where(mask)[0]
        if len(valid) == 0:
            continue
        # Diagonal fraction (guard against an all-zero attention row block)
        denom = np.sum(attn[b, valid, :][:, valid])
        if denom > 1e-12:
            diag = np.sum(attn[b, valid, valid]) / denom
            diag_fracs.append(diag)
        # For each closing bracket
        close_positions = np.where((batch.input_ids[b] == VOCAB["CLOSE"]) & mask)[0]
        for pos in close_positions:
            match_pos = batch.matching_open_pos[b, pos]
            if match_pos >= 0:
                # Matching accuracy: is max attention on the matching open?
                attn_to_valid = attn[b, pos, :].copy()  # copy: do not mutate the caller's array
                attn_to_valid[~mask] = -1
                max_pos = int(np.argmax(attn_to_valid))
                matching_accs.append(1.0 if max_pos == match_pos else 0.0)
            # Depth correlation: correlate attention to all open positions with their nesting depth
            open_positions = np.where((batch.input_ids[b] == VOCAB["OPEN"]) & mask)[0]
            if len(open_positions) > 1:
                attn_to_opens = attn[b, pos, open_positions]
                true_depths = batch.open_depth[b, open_positions].astype(float)
                if np.std(attn_to_opens) > 1e-8 and np.std(true_depths) > 1e-8:
                    r = np.corrcoef(attn_to_opens, true_depths)[0, 1]
                    if not np.isnan(r):
                        depth_corrs.append(r)
    return {
        "matching_accuracy": float(np.mean(matching_accs)) if matching_accs else 0.0,
        "depth_corr": float(np.mean(depth_corrs)) if depth_corrs else 0.0,
        "diag_frac": float(np.mean(diag_fracs)) if diag_fracs else 0.0,
    }

def _linear_baseline_matching(batch: Batch) -> float:
    """Matching accuracy of a fixed head that attends uniformly to all prior open brackets.

    Under uniform attention every prior open ties, so the metric's argmax (which
    breaks ties toward the lowest index) always selects the earliest prior open.
    This is the no-mechanism reference, computed on the canonical batch so it
    tracks the seed/scale instead of being a stale constant.
    """
    accs = []
    positions = np.arange(batch.seq_len)
    for b in range(batch.input_ids.shape[0]):
        mask = batch.attention_mask[b]
        ids = batch.input_ids[b]
        close_positions = np.where((ids == VOCAB["CLOSE"]) & mask)[0]
        for pos in close_positions:
            match_pos = batch.matching_open_pos[b, pos]
            if match_pos < 0:
                continue
            prior_opens = np.where((ids == VOCAB["OPEN"]) & mask & (positions < pos))[0]
            if len(prior_opens) == 0:
                continue
            pred = int(prior_opens.min())  # uniform attn -> argmax picks lowest index
            accs.append(1.0 if pred == match_pos else 0.0)
    return float(np.mean(accs)) if accs else 0.0

def evaluate(model_fn) -> Dict[str, Any]:
    """Run model_fn on canonical test batch, return payload."""
    batch = generate(CANONICAL_SEED)
    out = model_fn(batch.input_ids, batch.attention_mask)
    attn_weights = out["attn_weights"]  # (batch, n_heads, seq_len, seq_len)
    if attn_weights.ndim != 4:
        raise ValueError(f"attn_weights must be 4D, got {attn_weights.shape}")
    batch_size, n_heads, seq_len, _ = attn_weights.shape
    if batch_size != batch.input_ids.shape[0] or seq_len != batch.seq_len:
        raise ValueError(f"Shape mismatch: batch {batch.input_ids.shape}, attn {attn_weights.shape}")
    per_head = []
    for h in range(n_heads):
        head_attn = attn_weights[:, h, :, :]  # (batch, seq_len, seq_len)
        metrics = _compute_head_metrics(head_attn, batch)
        per_head.append({"head": h, **metrics})
    best_match = max(p["matching_accuracy"] for p in per_head)
    best_corr = max(p["depth_corr"] for p in per_head)
    return {
        "version": 1,
        "canonical_seed": CANONICAL_SEED,
        "seq_len": batch.seq_len,
        "max_depth": batch.max_depth,
        "n_heads": n_heads,
        "n_layers": 1,  # attempt documents which layer
        "per_head": per_head,
        "aggregated": {
            "best_matching_accuracy": best_match,
            "best_depth_corr": best_corr,
            "linear_baseline_matching": _linear_baseline_matching(batch),
        },
    }

def random_model_fn():
    """Return a model_fn that outputs uniform attention over valid positions."""
    def _fn(input_ids: np.ndarray, attention_mask: np.ndarray) -> Dict[str, np.ndarray]:
        batch_size, seq_len = input_ids.shape
        # Uniform over valid keys for each query
        attn = np.zeros((batch_size, 1, seq_len, seq_len), dtype=np.float32)  # 1 head
        for b in range(batch_size):
            mask = attention_mask[b]
            valid = np.where(mask)[0]
            n_valid = len(valid)
            if n_valid > 0:
                attn[b, 0, valid[:, None], valid[None, :]] = 1.0 / n_valid
        return {"attn_weights": attn}
    return _fn