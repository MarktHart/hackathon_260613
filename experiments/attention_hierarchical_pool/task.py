import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class Batch:
    input_ids: np.ndarray          # (batch, seq_len) int32
    # Ground-truth hierarchical structure (same for all batches at canonical condition)
    chunk_size: int = 16
    num_chunks: int = 16
    superchunk_size: int = 4       # chunks per superchunk

# Canonical constants
SEQ_LEN = 256
NUM_LAYERS = 12
NUM_HEADS = 8
CHUNK_SIZE = 16
NUM_CHUNKS = 16
SUPERCHUNK_SIZE = 4                # 4 chunks = 64 tokens
BATCH_SIZE = 1

# Type for the model function attempts must provide
ModelFn = Callable[[np.ndarray, int, int], np.ndarray]


def generate(seed: int = 0) -> Batch:
    """Deterministic synthetic batch with hierarchical token structure.

    Token IDs encode position in hierarchy: token_id = chunk_idx * chunk_size + pos_in_chunk.
    The same batch is returned for any seed (fully fixed canonical condition).
    """
    # Fixed pattern: tokens 0-15 = chunk 0, 16-31 = chunk 1, etc.
    input_ids = np.arange(SEQ_LEN, dtype=np.int32).reshape(BATCH_SIZE, SEQ_LEN)
    return Batch(
        input_ids=input_ids,
        chunk_size=CHUNK_SIZE,
        num_chunks=NUM_CHUNKS,
        superchunk_size=SUPERCHUNK_SIZE
    )


def _chunk_of(pos: int, chunk_size: int) -> int:
    return pos // chunk_size


def _superchunk_of(pos: int, chunk_size: int, superchunk_size: int) -> int:
    chunk = pos // chunk_size
    return chunk // superchunk_size


def _compute_concentrations(
    attn: np.ndarray,                 # (seq_len, seq_len) — single head, single batch item
    chunk_size: int,
    num_chunks: int,
    superchunk_size: int
) -> tuple[float, float, float, float]:
    """Compute local, chunk, superchunk concentration and entropy for one attention matrix."""
    seq_len = attn.shape[0]
    assert seq_len == chunk_size * num_chunks

    local_mass = 0.0
    chunk_mass = 0.0
    superchunk_mass = 0.0
    total_entropy = 0.0

    for q in range(seq_len):
        q_chunk = _chunk_of(q, chunk_size)
        q_superchunk = _superchunk_of(q, chunk_size, superchunk_size)
        row = attn[q]

        # Local: same chunk, within ±2 positions (5-token window centered on query)
        local_mask = np.zeros(seq_len, dtype=bool)
        for k in range(max(0, q - 2), min(seq_len, q + 3)):
            if _chunk_of(k, chunk_size) == q_chunk:
                local_mask[k] = True
        local_mass += row[local_mask].sum()

        # Chunk: entire chunk containing query
        chunk_start = q_chunk * chunk_size
        chunk_end = chunk_start + chunk_size
        chunk_mass += row[chunk_start:chunk_end].sum()

        # Superchunk: 4 chunks containing query
        sc_start = q_superchunk * superchunk_size * chunk_size
        sc_end = sc_start + superchunk_size * chunk_size
        superchunk_mass += row[sc_start:sc_end].sum()

        # Entropy (nats)
        row_clipped = np.clip(row, 1e-12, 1.0)
        total_entropy += -(row_clipped * np.log(row_clipped)).sum()

    n_queries = seq_len
    return (
        float(local_mass / n_queries),
        float(chunk_mass / n_queries),
        float(superchunk_mass / n_queries),
        float(total_entropy / n_queries)
    )


def evaluate(model_fn: ModelFn) -> dict:
    """Run model_fn over all (layer, head) pairs, return payload for benchmark.score."""
    batch = generate()
    seq_len = batch.input_ids.shape[1]

    sweep = []
    for layer in range(NUM_LAYERS):
        for head in range(NUM_HEADS):
            # model_fn returns (batch, seq_len, seq_len); we take batch item 0
            attn = model_fn(batch.input_ids, layer, head)  # (1, seq_len, seq_len)
            attn = attn[0]  # (seq_len, seq_len)

            # Validate
            if attn.shape != (seq_len, seq_len):
                raise ValueError(f"model_fn returned shape {attn.shape}, expected ({seq_len}, {seq_len})")
            if not np.allclose(attn.sum(axis=1), 1.0, atol=1e-4):
                raise ValueError("Attention rows must sum to 1")

            local_c, chunk_c, superchunk_c, entropy = _compute_concentrations(
                attn, CHUNK_SIZE, NUM_CHUNKS, SUPERCHUNK_SIZE
            )

            sweep.append({
                "layer": layer,
                "head": head,
                "local_concentration": local_c,
                "chunk_concentration": chunk_c,
                "superchunk_concentration": superchunk_c,
                "entropy": entropy
            })

    return {
        "version": 1,
        "seq_len": SEQ_LEN,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "chunk_size": CHUNK_SIZE,
        "num_chunks": NUM_CHUNKS,
        "sweep": sweep
    }


def random_model_fn() -> ModelFn:
    """Returns a dummy model_fn that outputs uniform attention for testing."""
    def _fn(input_ids: np.ndarray, layer_idx: int, head_idx: int) -> np.ndarray:
        batch_size, seq_len = input_ids.shape
        # Uniform attention: each query attends equally to all keys
        attn = np.ones((batch_size, seq_len, seq_len), dtype=np.float32) / seq_len
        return attn
    return _fn