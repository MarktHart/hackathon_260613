"""attention_argmin / pass_3 — hand-built argmin attention head.

Approach: HAND_BUILT (no training). The claim is mechanistic and precise:

    an attention head implements argmin over per-position scalar values by
    embedding the scalar value into an extra key channel and pointing the
    query at that channel with a NEGATIVE weight (-beta). The attention
    logit at position i then equals exactly  -beta * value_i, so the softmax
    — which concentrates on the *largest* logit — concentrates on the
    *smallest* value, i.e. the argmin. `beta` is the inverse temperature
    that controls how sharply.

This is the smallest delta from `base_model.py`: a single attention head, no
MLP, with W_Q / W_K hand-set so the effective logit is -beta * value. The
dot-product score is a real matmul executed on the GPU.

main.py:
  * evaluates the canonical head (beta = BETA) and records the benchmark;
  * sweeps beta to show the mechanism is causal — beta = 0 collapses the head
    to the uniform no-mechanism strawman, and raising beta sharpens the argmin.
    That sweep is saved for the demo (sharpness vs beta).
"""
import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

# The pipeline guarantees a visible CUDA device. No CPU fallback (it fails the
# GPU guard).
DEVICE = "cuda"

task = load_task(__file__)

# Inverse temperature for the canonical, scored head. Large enough that the
# argmin dominates at the canonical gap while staying numerically tame.
BETA = 20.0

# Beta sweep used for the causal / demo evidence (0.0 == ablated mechanism).
BETA_SWEEP = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 32.0]


def make_argmin_head(beta: float):
    """Return a model_fn implementing the hand-built argmin attention head.

    The head augments each 32-dim key with a 33rd channel holding the scalar
    value, and uses a hand-set query that is zero on the 32 random key dims and
    -beta on the value channel. Hence:

        logit_i = q_eff . k_aug_i = -beta * value_i

    softmax(logit) puts mass on the minimum value. beta = 0 -> uniform.
    """
    beta_t = torch.tensor(float(beta), dtype=torch.float32, device=DEVICE)

    def model_fn(keys: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
        k = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)      # (L, 32)
        v = torch.as_tensor(values, dtype=torch.float32, device=DEVICE)    # (L,)

        # Augment the key with the scalar value as an extra channel.
        k_aug = torch.cat([k, v[:, None]], dim=1)                          # (L, 33)

        # Hand-set effective query: ignore the random key dims, weight the
        # value channel by -beta. This is the W_Q/W_K specialisation.
        q_eff = torch.zeros(k_aug.shape[1], dtype=torch.float32, device=DEVICE)
        q_eff[-1] = -beta_t

        # Real attention score (matmul on GPU): logit_i = -beta * value_i.
        logits = k_aug @ q_eff                                             # (L,)
        attn = torch.softmax(logits, dim=-1)                              # (L,)
        return attn.detach().cpu().numpy()

    return model_fn


def main():
    run_dir = results_dir(__file__)

    # ---- canonical scored payload (beta = BETA) ----
    payload = task.evaluate(make_argmin_head(BETA))
    record_benchmark(__file__, run_dir, payload)

    # ---- beta sweep: causal evidence that the -beta*value channel is the
    #      mechanism. beta = 0 == ablated head == uniform strawman. ----
    # sharpness == attn_at_min * seq_len, exactly the goal's headline metric.
    def _sharpness(rec):
        return float(rec["attn_at_min"]) * int(rec["seq_len"])

    sweep_rows = []
    for beta in BETA_SWEEP:
        p = task.evaluate(make_argmin_head(beta))
        canon = p["canonical"]
        per_gap = {f"{rec['gap']:.2f}": _sharpness(rec) for rec in p["sweep"]}
        sweep_rows.append(
            {
                "beta": float(beta),
                "sharpness_canonical": _sharpness(canon),
                "accuracy_canonical": float(canon["argmax_correct"]),
                "attn_at_min_canonical": float(canon["attn_at_min"]),
                "sharpness_per_gap": per_gap,
            }
        )

    with open(run_dir / "beta_sweep.json", "w") as fh:
        json.dump(
            {
                "beta_canonical": BETA,
                "gaps": list(task.GAPS),
                "canonical_gap": task.CANONICAL_GAP,
                "seq_len": task.SEQ_LEN,
                "rows": sweep_rows,
            },
            fh,
            indent=2,
        )

    headline = _sharpness(payload["canonical"])
    print(f"[attention_argmin/pass_3] beta={BETA} "
          f"argmin_sharpness_canonical={headline:.3f} (uniform baseline = 1.0)")


if __name__ == "__main__":
    main()
