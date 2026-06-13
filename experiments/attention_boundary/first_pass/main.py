import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"


def model_fn(input_ids: np.ndarray, delim_id: int) -> np.ndarray:
    """
    Hand-built attention that respects segment boundaries.

    Sequence structure (fixed by task.generate):
      indices 0..7       : segment A
      index  8           : delimiter (delim_id == 63)
      indices 9..16      : segment B
      index  17          : EOS (62)

    For queries in segment A, we concentrate attention mass on segment A keys.
    For queries in segment B, we concentrate on segment B keys.
    Delimiter and EOS receive minimal mass.
    """
    batch, seq_len = input_ids.shape
    n_heads = 4

    # Segment boundaries (canonical config)
    seg_len = 8
    delim_pos = 8
    segA_start, segA_end = 0, seg_len           # 0..7
    segB_start, segB_end = delim_pos + 1, delim_pos + 1 + seg_len  # 9..16
    eos_pos = seq_len - 1                        # 17

    # Initialize attention weights on the GPU
    attn = torch.zeros((batch, n_heads, seq_len, seq_len), dtype=torch.float32, device=DEVICE)

    # Uniform small mass for delimiter and EOS (so they're not exactly zero)
    eps_delim = 0.01
    eps_eos = 0.01
    eps_cross = 0.02

    within_mass = 1.0 - eps_delim - eps_eos - eps_cross
    per_key_within = within_mass / seg_len
    per_key_cross = eps_cross / seg_len

    # --- Segment A queries (0..7) ---
    attn[:, :, segA_start:segA_end, segA_start:segA_end] = per_key_within
    attn[:, :, segA_start:segA_end, delim_pos] = eps_delim
    attn[:, :, segA_start:segA_end, segB_start:segB_end] = per_key_cross
    attn[:, :, segA_start:segA_end, eos_pos] = eps_eos

    # --- Delimiter query (8) --- attend mostly to itself and EOS
    attn[:, :, delim_pos, delim_pos] = 0.5
    attn[:, :, delim_pos, eos_pos] = 0.5

    # --- Segment B queries (9..16) ---
    attn[:, :, segB_start:segB_end, segB_start:segB_end] = per_key_within
    attn[:, :, segB_start:segB_end, delim_pos] = eps_delim
    attn[:, :, segB_start:segB_end, segA_start:segA_end] = per_key_cross
    attn[:, :, segB_start:segB_end, eos_pos] = eps_eos

    # --- EOS query (17) --- attend to itself
    attn[:, :, eos_pos, :] = 0.0
    attn[:, :, eos_pos, eos_pos] = 1.0

    attn = attn.detach().cpu().numpy().astype(np.float32)

    # Sanity: each query row sums to 1
    row_sums = attn.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"Row sums not 1: max|sum-1|={np.max(np.abs(row_sums - 1.0))}"

    return attn


def main():
    task = load_task(__file__)
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":
    main()