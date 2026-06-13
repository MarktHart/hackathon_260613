"""attention_longest_run / first_pass

Hand-built circuit (no training): measure the longest consecutive run of
attention weights above the canonical threshold (0.5), with a small
morphological-closing denoiser to repair single-position noise drop-outs.

All real compute runs in torch on CUDA (threshold, morphological close,
run-length scan). task.py hands NumPy arrays in and expects NumPy back.
"""
import json
import os

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU; no CPU fallback
THRESHOLD = 0.5

task = load_task(__file__)


# ----------------------------------------------------------------------------
# The circuit, expressed as torch ops on the GPU.
# ----------------------------------------------------------------------------
def _close(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Morphological closing along the sequence axis (dilate then erode).

    Bridges single-position gaps in a high-attention run caused by additive
    noise pushing one weight below threshold. mask: (B, H, S) of {0,1}.
    """
    pad = k // 2
    dil = torch.nn.functional.max_pool1d(mask, kernel_size=k, stride=1, padding=pad)
    ero = -torch.nn.functional.max_pool1d(-dil, kernel_size=k, stride=1, padding=pad)
    return ero


def _longest_run(mask: torch.Tensor) -> torch.Tensor:
    """Longest contiguous run of 1s along the seq axis. mask: (B, H, S)."""
    B, H, S = mask.shape
    run = torch.zeros((B, H), device=mask.device)
    best = torch.zeros((B, H), device=mask.device)
    for t in range(S):
        run = (run + 1.0) * mask[:, :, t]   # reset to 0 wherever mask==0
        best = torch.maximum(best, run)
    return best


def make_model_fn(denoise: bool = True, threshold: float = THRESHOLD):
    def model_fn(tokens: np.ndarray, attention_weights: np.ndarray) -> np.ndarray:
        w = torch.as_tensor(attention_weights, dtype=torch.float32, device=DEVICE)  # (B,H,S)
        mask = (w > threshold).float()
        if denoise:
            mask = _close(mask, k=3)
        best = _longest_run(mask)
        return best.detach().cpu().numpy().astype(np.float32)  # (B, H)
    return model_fn


def _per_difficulty_mae(payload: dict) -> dict:
    out = {}
    for r in payload["sweep"]:
        out.setdefault(r["difficulty"], []).append(r["mae"])
    return {d: float(np.mean(v)) for d, v in out.items()}


def _per_L_mae(payload: dict, difficulty: float) -> dict:
    out = {}
    for r in payload["sweep"]:
        if abs(r["difficulty"] - difficulty) < 1e-6:
            out[r["run_length"]] = r["mae"]
    return out


def main():
    run_dir = results_dir(__file__)

    # Primary contribution: denoised longest-run measurement.
    fn_denoised = make_model_fn(denoise=True)
    payload = task.evaluate(fn_denoised)
    record_benchmark(__file__, run_dir, payload)

    # Strawman: raw threshold + longest run, no denoising. Same condition.
    fn_raw = make_model_fn(denoise=False)
    payload_raw = task.evaluate(fn_raw)

    # ---- comparison artefact (denoised vs raw vs predict-the-mean baseline) ----
    Ls = sorted({r["run_length"] for r in payload["sweep"]})
    mean_L = float(np.mean(Ls))
    baseline_mae = float(np.mean([abs(L - mean_L) for L in Ls]))

    comparison = {
        "difficulties": sorted({r["difficulty"] for r in payload["sweep"]}),
        "run_lengths": Ls,
        "mae_denoised_by_d": _per_difficulty_mae(payload),
        "mae_raw_by_d": _per_difficulty_mae(payload_raw),
        "baseline_mae": baseline_mae,
        "mae_denoised_by_L_d0p5": _per_L_mae(payload, 0.5),
        "mae_raw_by_L_d0p5": _per_L_mae(payload_raw, 0.5),
        "threshold": THRESHOLD,
    }
    with open(os.path.join(run_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    # ---- sample artefact for the Demo tab ----
    batch = task.generate(seed=0)
    diffs = batch.difficulty_per_head.tolist()
    sample_Ls = [1, 3, 5, 8, 12, 16]
    samples = []
    for L in sample_Ls:
        idx = int(np.argmax(batch.run_length_per_sample == L))
        tokens = batch.tokens[idx]
        start = int(np.argmax(tokens == 0))  # first target-token position
        samples.append({
            "run_length": int(L),
            "start": start,
            "weights": batch.attention_weights[idx].tolist(),   # (n_heads, seq_len)
            "difficulty_per_head": diffs,
        })
    with open(os.path.join(run_dir, "samples.json"), "w") as f:
        json.dump({"seq_len": int(batch.tokens.shape[1]),
                   "threshold": THRESHOLD,
                   "samples": samples}, f)

    sc = _per_difficulty_mae(payload)
    print(f"[done] denoised MAE by d: { {k: round(v,3) for k,v in sc.items()} }")
    print(f"[done] baseline MAE: {baseline_mae:.3f}  ->  {run_dir}")


if __name__ == "__main__":
    main()
