"""attention_or / pass_4 — OR as a log-sum-exp soft-maximum attention circuit.

A *single uniform* attention circuit (no branching on query identity) in which
the logical-OR / max-pooling behaviour emerges from a log-sum-exp soft maximum
— the same operation that lives inside softmax attention.

For any query q and key matrix K (channels = the keys themselves):

    s[j] = (1/beta) * logsumexp_c( log_softmax_c(gamma * (q . k_c)) + beta * (k_c . k_j) )

    * gamma * (q . k_c)   -- query -> channel gate (first-order attention):
                             which key-directions does the query express?
    * beta  * (k_c . k_j) -- channel -> key footprint (key-key attention)
    * logsumexp / beta    -- a SOFT MAXIMUM over the gated channels

For a single feature query q_A, only channel A is gated on, so the readout is
the ordinary score q_A . K. For the balanced superposition q_AB both channels
gate on equally, so the readout becomes max(q_A.k_j, q_B.k_j) = OR. The max is
NOT looked up; it is produced by the soft-max as beta grows.

The exact same callable handles q_A, q_B and q_AB — the difference is entirely
in the data-dependent gate, never in a Python branch.
"""

import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU; no CPU fallback
BETA = 40.0      # soft-max sharpness (the OR temperature)
GAMMA = 100.0    # channel-gate sharpness (noise suppression; sharpness-invariant)

task = load_task(__file__)


# ----------------------------------------------------------------------------
# The mechanism (and its ablations) — all real GPU compute on cuda
# ----------------------------------------------------------------------------
def make_or_fn(beta: float, gamma: float):
    """Log-sum-exp soft-maximum attention. Uniform over every query."""
    def fn(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        q = torch.as_tensor(query, dtype=torch.float32, device=DEVICE)   # (d,)
        K = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)    # (d, n)
        G = K.t() @ K                                  # (n_chan, n_key)  k_c . k_j
        qk = q @ K                                     # (n_chan,)        q . k_c
        log_g = torch.log_softmax(gamma * qk, dim=0)   # channel gate (log space)
        M = log_g.unsqueeze(1) + beta * G              # (n_chan, n_key)
        s = torch.logsumexp(M, dim=0) / beta           # soft-max readout (n_key,)
        return s.detach().cpu().numpy()
    return fn


def plain_linear_fn(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """STRAWMAN: ordinary linear attention q . K (no max-pooling)."""
    q = torch.as_tensor(query, dtype=torch.float32, device=DEVICE)
    K = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)
    return (q @ K).detach().cpu().numpy()


def make_avg_fn(gamma: float):
    """ABLATION: replace the soft-max readout with a plain weighted average
    (the beta -> 0 limit). Kills the max; OR collapses to a blend."""
    def fn(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        q = torch.as_tensor(query, dtype=torch.float32, device=DEVICE)
        K = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)
        G = K.t() @ K
        qk = q @ K
        g = torch.softmax(gamma * qk, dim=0)           # (n_chan,)
        s = g @ G                                       # weighted avg of footprints
        return s.detach().cpu().numpy()
    return fn


# ----------------------------------------------------------------------------
# Metric helpers (mirror benchmark.py formulas, applied to a sweep payload)
# ----------------------------------------------------------------------------
def _safe_div(num, den):
    return float(num / den) if den > 0 else 0.0


def sharpness(rec):
    return _safe_div(min(rec["s_AB_at_A"], rec["s_AB_at_B"]),
                     max(rec["s_A_at_A"], rec["s_B_at_B"]))


def leakage(rec):
    return _safe_div(rec["s_AB_noise_max"],
                     max(rec["s_AB_at_A"], rec["s_AB_at_B"]))


def canon(payload):
    return payload["sweep"][0]  # canonical = cos(q_A, q_B) = 0.0


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
def main():
    or_fn = make_or_fn(BETA, GAMMA)

    # 1) Official benchmark payload for THIS attempt's contribution.
    payload = task.evaluate(or_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    # 2) Extra analysis for the Demo tab -------------------------------------
    lin_payload = task.evaluate(plain_linear_fn)

    # 2a) sharpness / leakage vs cos: OR circuit vs plain-linear strawman
    main_sweep = []
    for orr, lin in zip(payload["sweep"], lin_payload["sweep"]):
        main_sweep.append({
            "cos": orr["cos"],
            "or_sharpness": sharpness(orr),
            "plain_linear_sharpness": sharpness(lin),
            # ideal linear superposition value sqrt((1+cos)/2) for reference
            "linear_ideal": float(np.sqrt((1.0 + orr["cos"]) / 2.0)),
            "or_noise_leakage": leakage(orr),
        })

    # 2b) beta sweep at the canonical anchor: the soft-max temperature IS
    #     the OR knob (averaging -> hard max).
    beta_curve = []
    for b in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]:
        p = task.evaluate(make_or_fn(b, GAMMA))
        beta_curve.append({
            "beta": b,
            "sharpness": sharpness(canon(p)),
            "noise_leakage": leakage(canon(p)),
        })

    # 2c) component ablations at the canonical anchor
    p_avg = task.evaluate(make_avg_fn(GAMMA))           # no soft-max (beta->0)
    p_nogate = task.evaluate(make_or_fn(BETA, 0.0))     # no gate (gamma=0)
    ablation = {
        "full":         {"sharpness": sharpness(canon(payload)),
                         "noise_leakage": leakage(canon(payload))},
        "no_softmax":   {"sharpness": sharpness(canon(p_avg)),
                         "noise_leakage": leakage(canon(p_avg))},
        "no_gate":      {"sharpness": sharpness(canon(p_nogate)),
                         "noise_leakage": leakage(canon(p_nogate))},
        "plain_linear": {"sharpness": sharpness(canon(lin_payload)),
                         "noise_leakage": leakage(canon(lin_payload))},
    }

    analysis = {
        "beta": BETA,
        "gamma": GAMMA,
        "main_sweep": main_sweep,
        "beta_curve": beta_curve,
        "ablation": ablation,
    }
    (run_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))

    print(f"or_sharpness_canonical   = {sharpness(canon(payload)):.4f}")
    print(f"plain_linear (strawman)  = {sharpness(canon(lin_payload)):.4f}")
    print(f"no_softmax  (beta->0)    = {ablation['no_softmax']['sharpness']:.4f}")
    print(f"no_gate     (gamma=0)    leakage = {ablation['no_gate']['noise_leakage']:.4f}")
    print(f"artefacts -> {run_dir}")


if __name__ == "__main__":
    main()
