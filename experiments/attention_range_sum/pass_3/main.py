"""attention_range_sum / pass_3 — hand-built single-head range-sum circuit.

Mechanism (a minimal delta from experiments/base_model.py):
  * ONE attention head, NO MLP, NO unembed.
  * Position p carries a one-hot positional key  k_p = e_p  (the rows of I_64).
  * A query encodes the interval [start, end):  q = M * sum_{p in window} e_p.
  * scores_p = q . k_p = M  if p in [start,end) else 0.
        softmax over positions  ->  attn_p ~= 1/k on the window, ~0 outside.
  * The value carried by position p is the *token value* itself: v_p = input_ids[p].
        attention output = sum_p attn_p v_p = mean(window).
  * A length-scaled readout multiplies by the (known) window length k:
        prediction = mean(window) * k = sum(window).

So the head SELECTS the window with a genuine Q.K softmax and AGGREGATES the
values; the only non-attention step is the scalar *k readout. Everything runs
as float32 torch tensors on CUDA.

main.py also evaluates two ablations (kept out of the official payload, saved to
ablation.json) to demonstrate the head actually uses this circuit:
  * no_selection  — query is uniform (attends ALL 64 positions) -> predicts the
                    global mean * k, which collapses onto the constant baseline.
  * no_scaling    — drop the *k readout -> predicts the mean, not the sum.
"""

import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

DEVICE = "cuda"  # guaranteed by the pipeline; do NOT fall back to CPU
device = torch.device(DEVICE)

SEQ_LEN = 64
GAIN = 30.0  # softmax logit gain: e^30 ~ 1e13 -> leak onto out-of-window ~ 0

# Hand-set head weights, materialised on the GPU once.
POS = torch.arange(SEQ_LEN, device=device)          # positions 0..63
KEYS = torch.eye(SEQ_LEN, device=device)            # k_p = e_p  (one-hot keys)


def _attention(start: int, end: int, *, select: bool) -> torch.Tensor:
    """Softmax attention weights over the 64 positions for window [start, end)."""
    if select:
        indicator = ((POS >= start) & (POS < end)).to(torch.float32)
        query = GAIN * indicator                    # interval-encoding query
    else:
        query = torch.zeros(SEQ_LEN, device=device)  # ablation: uniform query
    scores = KEYS @ query                            # scores_p = q . k_p
    return torch.softmax(scores, dim=0)


def make_model_fn(*, select: bool = True, scale: bool = True):
    """Build a model_fn(input_ids, start, end) -> float on the GPU."""

    def model_fn(input_ids, start: int, end: int) -> float:
        start, end = int(start), int(end)
        k = max(end - start, 1)
        vals = torch.as_tensor(input_ids, dtype=torch.float32, device=device)
        attn = _attention(start, end, select=select)
        mean = torch.dot(attn, vals)                 # mean of selected window
        pred = mean * (float(k) if scale else 1.0)   # length-scaled readout
        return float(pred.detach().cpu().item())

    return model_fn


def _mse_by_k(payload: dict) -> dict:
    out = {}
    for rec in payload["sweep"]:
        p = np.asarray(rec["predictions"], dtype=np.float64)
        t = np.asarray(rec["targets"], dtype=np.float64)
        out[int(rec["range_len"])] = float(np.mean((p - t) ** 2))
    return out


def _var_by_k(payload: dict) -> dict:
    out = {}
    for rec in payload["sweep"]:
        t = np.asarray(rec["targets"], dtype=np.float64)
        out[int(rec["range_len"])] = float(np.var(t))
    return out


def main() -> None:
    run_dir = results_dir(__file__)

    # ---- official mechanism ----
    full_fn = make_model_fn(select=True, scale=True)
    payload = task.evaluate(full_fn)
    record_benchmark(__file__, run_dir, payload)

    # ---- ablations (faithfulness evidence) ----
    abl_no_select = task.evaluate(make_model_fn(select=False, scale=True))
    abl_no_scale = task.evaluate(make_model_fn(select=True, scale=False))

    range_lens = [rec["range_len"] for rec in payload["sweep"]]
    ablation = {
        "range_lens": range_lens,
        "full": _mse_by_k(payload),
        "no_selection": _mse_by_k(abl_no_select),
        "no_scaling": _mse_by_k(abl_no_scale),
        "baseline": _var_by_k(payload),  # constant-predictor floor = target variance
    }
    (run_dir / "ablation.json").write_text(json.dumps(ablation, indent=2))

    # console summary
    full = ablation["full"]
    base = ablation["baseline"]
    print("range_sum pass_3 — MSE by window length k")
    for k in range_lens:
        print(f"  k={k:2d}  full={full[k]:.3e}  baseline(var)={base[k]:.3e}")


if __name__ == "__main__":
    main()
