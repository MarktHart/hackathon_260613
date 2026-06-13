from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class Batch:
    src_ids: np.ndarray          # (batch, seq_len)
    tgt_ids: np.ndarray          # (batch, seq_len)
    true_perm: np.ndarray        # (batch, seq_len) — true_perm[b, tgt_pos] = src_pos
    perm_type: np.ndarray        # (batch,) — 0=swap, 1=rotation, 2=random


def generate(seed: int = 0) -> Batch:
    rng = np.random.default_rng(seed)
    seq_len = 8
    vocab_size = 50
    batch_size = 500

    # Generate source sequences
    src_ids = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)

    # Permutation types: 0=swap, 1=rotation, 2=random
    perm_types = rng.integers(0, 3, size=batch_size)
    true_perm = np.zeros((batch_size, seq_len), dtype=np.int32)
    tgt_ids = np.zeros_like(src_ids)

    for b in range(batch_size):
        ptype = perm_types[b]
        src = src_ids[b]

        if ptype == 0:  # single random swap
            perm = np.arange(seq_len)
            i, j = rng.choice(seq_len, size=2, replace=False)
            perm[i], perm[j] = perm[j], perm[i]
        elif ptype == 1:  # rotation by random offset
            offset = rng.integers(1, seq_len)
            perm = (np.arange(seq_len) + offset) % seq_len
        else:  # random permutation
            perm = rng.permutation(seq_len)

        true_perm[b] = perm
        tgt_ids[b] = src[perm]

    return Batch(
        src_ids=src_ids,
        tgt_ids=tgt_ids,
        true_perm=true_perm,
        perm_type=perm_types,
    )


def evaluate(model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> dict:
    batch = generate(seed=0)  # canonical seed
    src_ids = batch.src_ids
    tgt_ids = batch.tgt_ids
    true_perm = batch.true_perm
    perm_type = batch.perm_type

    # model_fn returns (batch, n_heads, tgt_len, src_len)
    attn = model_fn(src_ids, tgt_ids)
    batch_size, n_heads, tgt_len, src_len = attn.shape
    assert tgt_len == src_len == 8
    assert batch_size == 500
    seq_len = src_len

    # Compute alignment per head, per permutation type
    perm_type_names = ["swap", "rotation", "random"]
    sweep = []

    for ptype_idx, ptype_name in enumerate(perm_type_names):
        mask = (perm_type == ptype_idx)
        if not mask.any():
            continue

        # attn for this perm type: (n_examples, n_heads, tgt_len, src_len)
        attn_sub = attn[mask]
        perm_sub = true_perm[mask]  # (n_examples, seq_len)
        n_ex = attn_sub.shape[0]

        head_alignments = []
        for h in range(n_heads):
            # For each target position, get attention on true source position
            alignment_per_pos = []
            for tgt_pos in range(tgt_len):
                true_src_pos = perm_sub[:, tgt_pos]  # (n_ex,)
                # Gather attention weights at true source positions
                attn_on_true = attn_sub[np.arange(n_ex), h, tgt_pos, true_src_pos]
                alignment_per_pos.append(float(attn_on_true.mean()))

            mean_alignment = float(np.mean(alignment_per_pos))
            max_alignment = float(np.max(alignment_per_pos))

            head_alignments.append({
                "head_idx": h,
                "mean_alignment": mean_alignment,
                "max_alignment": max_alignment,
                "alignment_per_pos": alignment_per_pos,
            })

        sweep.append({
            "perm_type": ptype_name,
            "head_alignments": head_alignments,
            "layer_idx": 0,
        })

    # Random baseline: uniform attention -> 1/seq_len
    random_baseline = {}
    for ptype_name in perm_type_names:
        random_baseline[ptype_name] = {
            "mean_alignment": 1.0 / seq_len,
            "max_alignment": 1.0 / seq_len,
            "alignment_per_pos": [1.0 / seq_len] * seq_len,
        }

    return {
        "version": 1,
        "config": {
            "seq_len": seq_len,
            "vocab_size": 50,
            "batch_size": batch_size,
            "perm_types": perm_type_names,
            "n_heads": n_heads,
            "n_layers": 1,
        },
        "sweep": sweep,
        "random_baseline": random_baseline,
    }


def random_model_fn() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Returns a model_fn that outputs uniform attention (1/seq_len)."""
    def _fn(src_ids: np.ndarray, tgt_ids: np.ndarray) -> np.ndarray:
        batch_size = src_ids.shape[0]
        seq_len = src_ids.shape[1]
        # Return uniform attention: (batch, 8 heads, seq_len, seq_len) with 1/seq_len
        return np.full((batch_size, 8, seq_len, seq_len), 1.0 / seq_len, dtype=np.float32)
    return _fn