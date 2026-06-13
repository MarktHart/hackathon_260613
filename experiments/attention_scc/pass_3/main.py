"""attention_scc / pass_3 — hand-built single attention head.

Hypothesis
----------
A single attention head already has the *geometric* capacity to resolve up to
rho = K/d = 4 superimposed unit keys at 10 dB SNR. The thing that destroys that
capacity in vanilla attention is the softmax TEMPERATURE, not the geometry.

The query is Q' = K_target + n, n ~ N(0, sigma^2 I), sigma^2 = 1/(d*10^(SNR/10)),
then renormalised. Under that Gaussian noise model the posterior over which key
is the target is

    P(target = i | Q) ∝ exp( ||Q'|| * (Q · K_i) / sigma^2 ).

So the Bayes-optimal head is plain dot-product attention with a *hand-set*
inverse temperature  beta = ||Q'|| / sigma^2  — derived from the noise model,
NOT learned. The standard 1/sqrt(d) scaling is ~5000x too small here and
collapses the logit gap to chance.

This file:
  * scores the canonical payload with the beta-softmax head (the submission),
  * saves diagnostics for the app: a temperature sweep (chance -> perfect),
    an ablation table (knock out exp / temperature / query), and the
    target-vs-best-distractor logit gap that explains *why* it works.
Everything runs in torch on cuda.
"""
import math
import os
import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

DEVICE = "cuda"  # guaranteed by the pipeline; no CPU fallback.

# ----------------------------------------------------------------------
# Constants of the canonical condition (from the goal README / task.py).
# ----------------------------------------------------------------------
D = 64
SNR_DB = 10.0
SIGMA2 = 1.0 / (D * 10 ** (SNR_DB / 10.0))         # per-dim noise variance = 1/640
Q_NORM = math.sqrt(1.0 + D * SIGMA2)               # E||Q'|| = sqrt(1 + d*sigma^2)
BETA = Q_NORM / SIGMA2                              # Bayes-optimal inverse temperature ~= 671
SCALE_STD = 1.0 / math.sqrt(D)                      # the (wrong, here) vanilla scaling = 0.125


# ----------------------------------------------------------------------
# Heads — all do their real compute on the GPU.
# ----------------------------------------------------------------------
def softmax_head(scale: float):
    """Dot-product attention with a fixed inverse-temperature `scale`."""
    def head(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
        qt = torch.as_tensor(Q, dtype=torch.float32, device=DEVICE)   # [d]
        kt = torch.as_tensor(K, dtype=torch.float32, device=DEVICE)   # [K, d]
        logits = (kt @ qt) * scale                                    # [K]
        attn = torch.softmax(logits, dim=-1)                          # stable (max-subtract)
        return attn.detach().cpu().numpy()
    return head


def linear_head(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Ablate the exp: relu(Q·K) normalised. Same head, no softmax sharpening."""
    qt = torch.as_tensor(Q, dtype=torch.float32, device=DEVICE)
    kt = torch.as_tensor(K, dtype=torch.float32, device=DEVICE)
    w = torch.relu(kt @ qt)
    s = w.sum()
    if float(s) <= 0.0:
        n = kt.shape[0]
        return (torch.ones(n, device=DEVICE) / n).detach().cpu().numpy()
    return (w / s).detach().cpu().numpy()


def uniform_head(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Ablate the query: uniform attention (the 1/K chance baseline)."""
    qt = torch.as_tensor(Q, dtype=torch.float32, device=DEVICE)        # touch the GPU
    kt = torch.as_tensor(K, dtype=torch.float32, device=DEVICE)
    n = kt.shape[0]
    _ = qt.sum() + kt.sum()                                            # ensure real cuda work
    return (torch.ones(n, device=DEVICE) / n).detach().cpu().numpy()


METHOD = softmax_head(BETA)


# ----------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------
def trapz_norm(xs, ys):
    """Normalised trapezoidal AUC (same convention as benchmark.py)."""
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return float(ys[0])
    area = 0.0
    for i in range(n - 1):
        area += 0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i])
    width = xs[-1] - xs[0]
    if width <= 0:
        return float(ys[0])
    return float(area / width)


def sweep_xy(payload):
    rhos = [r["rho"] for r in payload["sweep"]]
    means = [r["target_attention_mean"] for r in payload["sweep"]]
    chances = [r["chance_level"] for r in payload["sweep"]]
    return rhos, means, chances


def attn_at_rho1(rhos, means):
    for r, m in zip(rhos, means):
        if abs(r - 1.0) < 1e-9:
            return m
    return float("nan")


# ----------------------------------------------------------------------
# 1) Score the submission (Bayes-optimal beta-softmax head).
# ----------------------------------------------------------------------
payload = task.evaluate(METHOD)
run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)

m_rhos, m_means, m_chance = sweep_xy(payload)
print(f"[pass_3] BETA={BETA:.1f}  scc_auc={trapz_norm(m_rhos, m_means):.4f}  "
      f"chance_auc={trapz_norm(m_rhos, m_chance):.4f}")

# ----------------------------------------------------------------------
# 2) Diagnostics for the app.
# ----------------------------------------------------------------------
# 2a) Temperature sweep: inverse-temperature scale -> capacity. Includes the
#     exact vanilla (1/sqrt(d)) and Bayes-optimal (BETA) points so markers land
#     on the curve.
scales = sorted(set(
    [SCALE_STD, BETA] + [float(10 ** e) for e in np.linspace(-1.2, 3.4, 14)]
))
temperature_sweep = []
for sc in scales:
    p = task.evaluate(softmax_head(sc))
    rs, ms, _ = sweep_xy(p)
    temperature_sweep.append({
        "scale": float(sc),
        "scc_auc": trapz_norm(rs, ms),
        "attn_rho1": attn_at_rho1(rs, ms),
        "per_rho": {str(r): float(m) for r, m in zip(rs, ms)},
    })

# 2b) Ablations: knock out one piece of the circuit at a time.
baseline_std_payload = task.evaluate(softmax_head(SCALE_STD))
ablations = []
for name, fn in [
    ("full (beta-softmax)", METHOD),
    ("ablate exp (relu/sum)", linear_head),
    ("ablate temp (1/sqrt(d) softmax)", softmax_head(SCALE_STD)),
    ("ablate query (uniform)", uniform_head),
]:
    p = task.evaluate(fn)
    rs, ms, cs = sweep_xy(p)
    ablations.append({
        "name": name,
        "scc_auc": trapz_norm(rs, ms),
        "attn_rho1": attn_at_rho1(rs, ms),
        "chance_auc": trapz_norm(rs, cs),
        "per_rho": {str(r): float(m) for r, m in zip(rs, ms)},
    })

# 2c) Logit-gap structure: target logit minus best-distractor logit. Positive
#     gap => target is the argmax => sharp softmax lands ~all mass on target.
batch = task.generate(0)
logit_gap = []
for rho in batch.rhos:
    inst = batch.instances[rho]
    K = int(len(inst[0][1]))
    gaps = []
    correct = 0
    for Q, K_mat, tidx in inst:
        qt = torch.as_tensor(Q, dtype=torch.float32, device=DEVICE)
        kt = torch.as_tensor(K_mat, dtype=torch.float32, device=DEVICE)
        logits = kt @ qt
        tl = logits[tidx]
        others = logits.clone()
        others[tidx] = float("-inf")
        mo = others.max()
        gap = float((tl - mo).item())
        gaps.append(gap)
        if gap > 0:
            correct += 1
    gaps = np.asarray(gaps, dtype=np.float64)
    logit_gap.append({
        "rho": float(rho),
        "K": K,
        "mean_gap": float(gaps.mean()),
        "std_gap": float(gaps.std()),
        "frac_target_argmax": correct / len(inst),
    })

diagnostics = {
    "d": D,
    "snr_db": SNR_DB,
    "beta_opt": float(BETA),
    "scale_std": float(SCALE_STD),
    "sigma2": float(SIGMA2),
    "method_sweep": payload["sweep"],
    "baseline_standard_sweep": baseline_std_payload["sweep"],
    "temperature_sweep": temperature_sweep,
    "ablations": ablations,
    "logit_gap": logit_gap,
}

with open(os.path.join(str(run_dir), "diagnostics.json"), "w") as f:
    json.dump(diagnostics, f, indent=2)

print(f"[pass_3] wrote diagnostics to {run_dir}/diagnostics.json")
