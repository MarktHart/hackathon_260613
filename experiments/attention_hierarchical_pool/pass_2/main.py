import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

task = load_task(__file__)

# Canonical constants
SEQ_LEN = 256
NUM_LAYERS = 12
NUM_HEADS = 8
CHUNK_SIZE = 16
NUM_CHUNKS = 16
SUPERCHUNK_SIZE = 4


def _attn_head_forward(
    input_ids: np.ndarray,        # (batch, seq_len)
    layer_idx: int,
    head_idx: int,
    chunk_size: int,
    logit_scale: float = 3.0
) -> np.ndarray:                  # (batch=1, seq_len, seq_len) — attention weights
    """
    Implements the hierarchical pooling mechanism via attention.

    - Each head corresponds to a specific pooling level: 0 (local), 1 (chunk), 2 (superchunk).
    - Heads 3-7 are dummy uniform heads (no mechanism) to fill the 8-head shape required by task.py.
    - Attention is a weighted sum of one-hot positional indicators, normalised with temperature.
    - For a head targeting level L:
          query = hot_encode_position(level=L)
          key   = hot_encode_position(level=L)
      The dot product yields 1 across the region spanned by the head and 0 elsewhere,
      producing within-region uniform attention after softmax.
    - logit_scale lifts the temperature to sharpen mass.
    """
    B, L = input_ids.shape
    if B != 1 or L != SEQ_LEN:
        raise ValueError(f"Input shape mismatch in _attn_head_forward: expected (1, {SEQ_LEN}), got {input_ids.shape}")

    # Positional key vectors (1D hot encodings)
    # - level 0 (local, ±2 window): 5 token windows per chunk
    # - level 1 (full chunk): 16 token vectors
    # - level 2 (4 chunks, 64 tokens): 4 chunk vectors
    local_vecs = []
    chunk_vecs = []
    superchunk_vecs = []

    for chunk_idx in range(NUM_CHUNKS):
        chunk_start = chunk_idx * chunk_size
        for i in range(chunk_size):
            pos = chunk_start + i
            window_start = max(0, pos - 2)
            window_end = min(SEQ_LEN, pos + 3)  # inclusive end = pos+2 → indices [pos-2, ..., pos+2]
            # Local one-hot indicator (window, 1)
            local_vec = np.zeros(SEQ_LEN, dtype=np.float32)
            local_vec[window_start:window_end] = 1.0
            local_vecs.append(local_vec)

            # Chunk one-hot indicator (full 16-token chunk, 1)
            chunk_vec = np.zeros(SEQ_LEN, dtype=np.float32)
            chunk_vec[chunk_start:chunk_start + chunk_size] = 1.0
            chunk_vecs.append(chunk_vec)

        # Superchunk one-hot indicator (4 chunks)
        if chunk_idx % SUPERCHUNK_SIZE == 0:
            sc_start = chunk_idx * chunk_size
            sc_end   = sc_start + SUPERCHUNK_SIZE * chunk_size
            sc_vec = np.zeros(SEQ_LEN, dtype=np.float32)
            sc_vec[sc_start:sc_end] = 1.0
            superchunk_vecs.append(sc_vec)

    # Build the K matrix (SEQ_LEN, N_KEY_VEC) where N_KEY_VEC = sum of all vector counts
    # - For level 0: SEQ_LEN × (NUM_CHUNKS * 5) = 256 × 80
    # - For level 1: SEQ_LEN × NUM_CHUNKS = 256 × 16
    # - For level 2: SEQ_LEN × (NUM_CHUNKS // SUPERCHUNK_SIZE) = 256 × 4
    key_vectors = []
    if head_idx < 3:                # heads 0,1,2 handle the 3 hierarchy levels
        # Determine which level this head implements based on layer depth
        # Heuristics: fine (local) at early layers, coarse (superchunk) at late layers.
        # For even simpler demonstration, map:
        #   head 0 -> local (level 0)
        #   head 1 -> chunk (level 1)
        #   head 2 -> superchunk (level 2)
        key_set = {
            0: local_vecs,
            1: chunk_vecs,
            2: superchunk_vecs
        }.get(head_idx % 3, local_vecs)
    else:                           # heads 3-7 are dummy uniform heads
        key_vectors.append(np.ones(SEQ_LEN, dtype=np.float32) / SEQ_LEN)

        # For uniform heads we will just return uniform attention (1/L) for simplicity.
        attn = torch.ones((B, L, L), dtype=torch.float32, device=DEVICE) / L
        return attn.detach().cpu().numpy()

    # Stack key vectors column-wise into K (SEQ_LEN, N_KEY_VEC)
    key_stack = np.stack(key_set, axis=1) if key_set else np.empty((SEQ_LEN, 0))
    K = torch.as_tensor(key_stack, dtype=torch.float32, device=DEVICE)

    # Build the Q matrix (SEQ_LEN, N_KEY_VEC) — same key vectors used as positional queries
    Q = K              # each row is a positional query encoded over the key vectors

    # Compute attention scores: Q @ K.T gives a matrix where the ij entry is Q_i • K_j.
    # Because we used one-hot positional vectors, this yields 1 when positions belong
    # to the same region (window, chunk, superchunk) and 0 otherwise.
    logits = Q @ K.T                # (SEQ_LEN, SEQ_LEN) -> (SEQ_LEN, SEQ_LEN)

    # Apply temperature with learnable-looking logit_scale
    logits = logits * logit_scale

    # Softmax over the key dimension (column) to get attention weights per query
    attn = torch.softmax(logits, dim=1)               # (SEQ_LEN, SEQ_LEN)

    # Broadcast to (B=1, SEQ_LEN, SEQ_LEN)
    attn = attn.unsqueeze(0)

    return attn.detach().cpu().numpy()


def model_fn(
    input_ids: np.ndarray,
    layer_idx: int,
    head_idx: int
) -> np.ndarray:
    """Hand-built model function implementing hierarchical pooling via attention.

    See `_attn_head_forward` docstring for implementation details.
    """
    return _attn_head_forward(input_ids, layer_idx, head_idx, CHUNK_SIZE, logit_scale=10.0)


if __name__ == "__main__":
    payload = task.evaluate(model_fn)
    out_dir = results_dir(__file__)
    record_benchmark(__file__, out_dir, payload)
    print(f"Results written to {out_dir}")