"""Pass 3: fixed-geometry NOT head, ported to the current contract.

The goal's CURRENT contract (../task.py) is:

    model_fn(batch: Batch) -> {"attn_weights": np.ndarray (n_seq, seq_len, seq_len)}

layout [A-token, B-token, query, answer], per-sequence binary features feat_A
(attend) and feat_B (suppress), and known feature directions e_A / e_B plus the
head weights W_Q / W_K supplied on the Batch.

Original pass-3 idea (preserved): use the *known* fixed geometry to detect the
inhibitory marker and subtract its strength from the target score. Here the
marker direction is e_B and the attend direction is e_A. We project the query's
residual onto e_B to read off the inhibitory strength and subtract a scaled
version from the query->A-token attention score -- a genuine content-specific
inhibition. As the attend/suppress directions enter superposition (cos rises),
the detector degrades gracefully, matching the expected operating range.

All compute runs in torch on CUDA.
"""
from __future__ import annotations

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

_SUPPRESSION_SCALE = 8.0   # how strongly the detected marker suppresses the target
_ATTEND_GAIN = 4.0         # drive toward the A-token when feat_A is present


def make_model_fn():
    def model_fn(batch) -> dict:
        tokens = np.asarray(batch.tokens)
        n_seq, seq_len = tokens.shape
        A_POS, B_POS, QUERY_POS = 0, 1, 2

        feat_A = torch.as_tensor(np.asarray(batch.feat_A), dtype=torch.float32, device=DEVICE)
        feat_B = torch.as_tensor(np.asarray(batch.feat_B), dtype=torch.float32, device=DEVICE)
        e_A = torch.as_tensor(np.asarray(batch.e_A), dtype=torch.float32, device=DEVICE)
        e_B = torch.as_tensor(np.asarray(batch.e_B), dtype=torch.float32, device=DEVICE)
        W_Q = torch.as_tensor(np.asarray(batch.W_Q), dtype=torch.float32, device=DEVICE)
        W_K = torch.as_tensor(np.asarray(batch.W_K), dtype=torch.float32, device=DEVICE)

        d_model = e_A.shape[0]

        resid = torch.zeros((n_seq, seq_len, d_model), dtype=torch.float32, device=DEVICE)
        resid[:, A_POS, :] = e_A[None, :]
        resid[:, B_POS, :] = e_B[None, :]
        # Query carries the attend feature plus the raw suppress feature so the
        # fixed-geometry detector below can read the marker off it.
        resid[:, QUERY_POS, :] = feat_A[:, None] * e_A[None, :] + feat_B[:, None] * e_B[None, :]

        Q = resid @ W_Q
        K = resid @ W_K
        d_head = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(1, 2)) / (d_head ** 0.5)

        # Detect the inhibitory marker via the KNOWN direction e_B: project the
        # query residual onto e_B. marker_strength is ~feat_B here.
        q_resid = resid[:, QUERY_POS, :]              # (n_seq, d_model)
        marker_strength = q_resid @ e_B               # (n_seq,)

        base = _ATTEND_GAIN * feat_A
        scores[:, QUERY_POS, A_POS] = (
            scores[:, QUERY_POS, A_POS] + base - _SUPPRESSION_SCALE * marker_strength
        )

        attn = torch.softmax(scores, dim=-1)
        return {"attn_weights": attn.detach().cpu().numpy()}

    return model_fn


def main() -> None:
    task = load_task(__file__)
    model_fn = make_model_fn()
    payload = task.evaluate(model_fn)

    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    print("=== sweep ===")
    for rec in payload["sweep"]:
        print(
            f"  cos={rec['cos']:.1f}: not_sharpness={rec['not_sharpness']:.4f} "
            f"suppression_gap={rec['suppression_gap']:.4f} "
            f"attend_specificity={rec['attend_specificity']:.4f}"
        )


if __name__ == "__main__":
    main()
