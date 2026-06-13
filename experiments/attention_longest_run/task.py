from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class Batch:
    tokens: np.ndarray              # (batch, seq_len), int32
    attention_weights: np.ndarray   # (batch, n_heads, seq_len), float32
    true_run_lengths: np.ndarray    # (batch, n_heads), int32
    run_length_per_sample: np.ndarray  # (batch,), int32 — the L for each sample
    difficulty_per_head: np.ndarray    # (n_heads,), float32 — the d for each head
    threshold: float                # float, the canonical threshold (0.5)


ModelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def generate(seed: int = 0) -> Batch:
    # Fully deterministic; seed accepted for API compatibility but ignored.
    rng = np.random.default_rng(0)

    seq_len = 64
    vocab_size = 128
    target_token = 0
    run_lengths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
    difficulties = [0.3, 0.5, 0.7, 0.9]
    n_heads = 8
    samples_per_run_length = 128

    # Assign each head a fixed difficulty
    head_difficulties = np.array(difficulties * (n_heads // len(difficulties)), dtype=np.float32)
    assert len(head_difficulties) == n_heads

    all_tokens = []
    all_weights = []
    all_true_lengths = []
    all_run_length_per_sample = []
    all_difficulty_per_head = np.tile(head_difficulties, (samples_per_run_length * len(run_lengths), 1))

    for L in run_lengths:
        high_base = 0.5 + 0.5 * head_difficulties  # (n_heads,)
        low_base = 0.5 - 0.5 * head_difficulties

        for _ in range(samples_per_run_length):
            # Place a run of target_token of length L at a random position
            # such that the run fits entirely in the sequence
            max_start = seq_len - L
            start = int(rng.integers(0, max_start + 1))

            tokens = rng.integers(1, vocab_size, size=seq_len, dtype=np.int32)
            tokens[start:start + L] = target_token

            # For each head, generate attention weights over the sequence
            # Query position is the last token of the run (start + L - 1)
            # True attending positions: start ... start + L - 1 (the whole run)
            weights = np.zeros((n_heads, seq_len), dtype=np.float32)
            for h in range(n_heads):
                w = np.full(seq_len, low_base[h], dtype=np.float32)
                w[start:start + L] = high_base[h]
                # Add noise and clip
                w += rng.normal(0, 0.15, size=seq_len)
                w = np.clip(w, 0.0, 1.0)
                weights[h] = w

            # True longest run above threshold 0.5 for each head
            # Since we set high_base > 0.5 and low_base < 0.5 (for d>0), the run positions
            # should be above threshold and others below, but noise can flip some.
            # The *ground truth* run length is L (the implanted run).
            # But wait — the "true" longest run in the *noisy* weights might differ from L.
            # The spec says: "the true longest run for a query at the last target token is exactly L".
            # So we treat L as ground truth, not the noisy weights' actual longest run.
            true_lengths = np.full(n_heads, L, dtype=np.int32)

            all_tokens.append(tokens)
            all_weights.append(weights)
            all_true_lengths.append(true_lengths)
            all_run_length_per_sample.append(L)

    batch_size = len(all_tokens)
    tokens = np.stack(all_tokens, axis=0)                    # (batch, seq_len)
    attention_weights = np.stack(all_weights, axis=0)        # (batch, n_heads, seq_len)
    true_run_lengths = np.stack(all_true_lengths, axis=0)    # (batch, n_heads)
    run_length_per_sample = np.array(all_run_length_per_sample, dtype=np.int32)
    difficulty_per_head = head_difficulties                  # (n_heads,)

    return Batch(
        tokens=tokens,
        attention_weights=attention_weights,
        true_run_lengths=true_run_lengths,
        run_length_per_sample=run_length_per_sample,
        difficulty_per_head=difficulty_per_head,
        threshold=0.5,
    )


def random_model_fn() -> ModelFn:
    def _fn(tokens: np.ndarray, attention_weights: np.ndarray) -> np.ndarray:
        # Returns random predictions in the valid range [1, 16]
        batch, n_heads = tokens.shape[0], attention_weights.shape[1]
        rng = np.random.default_rng(42)  # Fixed seed for deterministic smoke test
        return rng.integers(1, 17, size=(batch, n_heads)).astype(np.float32)
    return _fn


def evaluate(model_fn: ModelFn) -> dict:
    batch = generate(seed=0)  # Canonical seed

    # Run the model function
    pred = model_fn(batch.tokens, batch.attention_weights)  # (batch, n_heads)

    # Validate prediction shape
    if pred.shape != batch.true_run_lengths.shape:
        raise ValueError(
            f"model_fn returned shape {pred.shape}, expected {batch.true_run_lengths.shape}"
        )

    # Compute per-slice metrics
    sweep = []
    unique_run_lengths = np.unique(batch.run_length_per_sample)
    unique_difficulties = np.unique(batch.difficulty_per_head)

    # Pearson correlation must be measured where the *true* value varies. Within
    # a single (run_length, difficulty) slice the true value is the constant L,
    # so a per-slice correlation is undefined (zero variance). We therefore
    # compute one correlation per difficulty, pooling predictions and truths
    # across all run lengths for the heads at that difficulty, and attach that
    # value to every record of the difficulty. Averaging over run lengths in
    # benchmark.score then reproduces it exactly.
    corr_by_difficulty = {}
    for d in unique_difficulties:
        head_mask = (batch.difficulty_per_head == d)
        pred_d = pred[:, head_mask].ravel()
        true_d = batch.true_run_lengths[:, head_mask].ravel()
        if len(pred_d) > 1 and np.std(pred_d) > 0 and np.std(true_d) > 0:
            corr_by_difficulty[float(d)] = float(np.corrcoef(pred_d, true_d)[0, 1])
        else:
            corr_by_difficulty[float(d)] = 0.0

    for L in unique_run_lengths:
        for d in unique_difficulties:
            # Mask for this (run_length, difficulty) slice
            sample_mask = (batch.run_length_per_sample == L)
            head_mask = (batch.difficulty_per_head == d)

            if not np.any(sample_mask) or not np.any(head_mask):
                continue

            # Extract predictions and truths for this slice
            pred_slice = pred[sample_mask][:, head_mask].ravel()
            true_slice = batch.true_run_lengths[sample_mask][:, head_mask].ravel()
            n_samples = len(pred_slice)

            # MAE
            mae = float(np.mean(np.abs(pred_slice - true_slice)))
            # RMSE
            rmse = float(np.sqrt(np.mean((pred_slice - true_slice) ** 2)))
            # Pearson correlation across run lengths at this difficulty
            corr = corr_by_difficulty[float(d)]

            sweep.append({
                "run_length": int(L),
                "difficulty": float(d),
                "mae": mae,
                "rmse": rmse,
                "correlation": corr,
                "n_samples": int(n_samples),
            })

    return {
        "version": 1,
        "canonical_threshold": batch.threshold,
        "sweep": sweep,
        "n_heads": int(batch.attention_weights.shape[1]),
        "seq_len": int(batch.tokens.shape[1]),
    }