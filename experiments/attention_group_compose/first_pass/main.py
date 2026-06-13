"""attention_group_compose / first_pass

Hand-built "snap-to-group" composer.

Hypothesis: noisy attention matrices that represent elements of the cyclic
group C_n compose according to the group law, NOT according to naive (softmax-
relaxed) matrix multiplication. So the right way to compose two noisy matrices
A, B is:

    1. Project each onto the nearest group element (a rotation k_a, k_b).
    2. Compose exactly in the group: k_c = (k_a + k_b) mod n.
    3. Return the clean permutation matrix P_{k_c}.

This is a hand-set circuit (no training). All real compute runs in torch on
CUDA: we score each input against the n rotation templates with an einsum and
take the argmax. The naive matmul baseline (A @ B) is computed by task.py for
comparison.
"""
import argparse
import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU; no CPU fallback


def _rotation_templates(n: int) -> torch.Tensor:
    """[n, n, n] tensor: R[k] is the permutation matrix for rotation by k in C_n."""
    R = torch.zeros((n, n, n), dtype=torch.float32, device=DEVICE)
    idx = torch.arange(n, device=DEVICE)
    for k in range(n):
        R[k, idx, (idx + k) % n] = 1.0
    return R


# Cache templates per n so we don't rebuild on every one of the 1000 calls.
_TEMPLATE_CACHE: dict[int, torch.Tensor] = {}


def _templates(n: int) -> torch.Tensor:
    if n not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[n] = _rotation_templates(n)
    return _TEMPLATE_CACHE[n]


def snap_compose_fn(attn_a: np.ndarray, attn_b: np.ndarray) -> np.ndarray:
    """Project A, B to nearest C_n rotation, compose in the group, return clean P."""
    n = attn_a.shape[0]
    R = _templates(n)  # [n, n, n] on cuda

    A = torch.as_tensor(attn_a, dtype=torch.float32, device=DEVICE)
    B = torch.as_tensor(attn_b, dtype=torch.float32, device=DEVICE)

    # Score each input against every rotation template: <A, R_k> = sum_ij A_ij R_k,ij.
    # Highest overlap == nearest group element (templates are 0/1 indicators).
    score_a = torch.einsum("ij,kij->k", A, R)
    score_b = torch.einsum("ij,kij->k", B, R)
    k_a = int(torch.argmax(score_a).item())
    k_b = int(torch.argmax(score_b).item())

    k_c = (k_a + k_b) % n
    out = R[k_c]  # exact clean permutation for the composed rotation
    return out.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()

    task = load_task(__file__)

    payload = task.evaluate(snap_compose_fn)

    run_dir = results_dir(__file__)

    # Save sweep for the Gradio app.
    sweep = payload["sweep"]
    with open(run_dir / "sweep.json", "w") as f:
        json.dump(sweep, f, indent=2)

    # Build a couple of concrete demo examples (one clean, one canonical noise)
    # so the Demo tab can show A, B, predicted C and true C side by side.
    batch = task.generate(seed=0)
    wanted = {0.0: None, 20.0: None}
    for A, B, true_comp, noise in batch.queries:
        if noise in wanted and wanted[noise] is None:
            pred = snap_compose_fn(A, B)
            baseline = A @ B
            wanted[noise] = {
                "noise_level": noise,
                "A": A.tolist(),
                "B": B.tolist(),
                "true": true_comp.tolist(),
                "pred": pred.tolist(),
                "baseline": baseline.tolist(),
            }
        if all(v is not None for v in wanted.values()):
            break
    demo_examples = {f"{k:.1f}": v for k, v in wanted.items() if v is not None}
    with open(run_dir / "demo_examples.json", "w") as f:
        json.dump(demo_examples, f, indent=2)

    record_benchmark(__file__, run_dir, payload)

    # Console summary
    print("noise  method_err  baseline_err")
    for rec in sweep:
        print(f"{rec['noise_level']:5.1f}  {rec['frobenius_error']:.4f}      "
              f"{rec['linear_baseline_error']:.4f}")


if __name__ == "__main__":
    main()
