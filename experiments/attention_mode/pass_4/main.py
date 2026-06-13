"""Hand-built feature-prototype attention-mode classifier (pass_4).

Approach (hand_built, no learning)
----------------------------------
Each head's (L, L) attention matrix is reduced to **five human-named scalar
features** that directly read the geometry the five modes are defined by:

    f_key0  = mean_i A[i, 0]          # mass on the fixed anchor key (positional)
    f_diag  = mean_i A[i, i]          # mass on the diagonal          (diagonal)
    f_next  = mean_i A[i, i+1]        # mass on the +1 band           (induction)
    f_prev  = mean_i A[i, i-1]        # mass on the -1 band           (previous_token)
    f_peak  = mean_i max_j A[i, j]    # row peakiness (low <=> uniform)

The five mode prototypes are derived *in closed form* from the very clean
templates `task.generate` uses (no fitting on the sweep). Classification is
nearest-prototype in this 5-D feature space, turned into a probability via a
softmax over the negative squared distance. Everything runs on CUDA.

A deliberately weakened **strawman** (drop f_key0 and f_peak, keep only the
three diagonal bands) is also computed and saved so the demo can show the
positional/uniform pair collapsing exactly when the disambiguating features
are removed.

model_fn contract (from task.py)
    input  : (n_heads, L, L) float32, rows sum to 1
    output : (n_heads, N_MODES) float32, rows sum to 1, in MODES order
"""

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

task = load_task(__file__)
MODES = list(task.MODES)            # positional, uniform, diagonal, induction, previous_token
N_MODES = len(MODES)
L = task.CANONICAL_L
BETA = 12.0                         # softmax sharpness on -distance


# ----------------------------------------------------------------------
# Clean templates (identical to task.py) -> used only to derive prototypes
# ----------------------------------------------------------------------
def _positional(L, anchor=0):
    p = np.zeros((L, L), np.float32); p[:, anchor % L] = 1.0; return p

def _uniform(L):
    return np.full((L, L), 1.0 / L, np.float32)

def _diagonal(L):
    return np.eye(L, dtype=np.float32)

def _induction(L):
    p = np.zeros((L, L), np.float32)
    for i in range(L):
        p[i, (i + 1) % L] = 1.0
    return p

def _previous(L):
    p = np.zeros((L, L), np.float32)
    for i in range(L):
        p[i, (i - 1) % L] = 1.0
    return p

_BUILD = {
    "positional": _positional,
    "uniform": _uniform,
    "diagonal": _diagonal,
    "induction": _induction,
    "previous_token": _previous,
}


# ----------------------------------------------------------------------
# Feature extraction (GPU)
# ----------------------------------------------------------------------
def _features(A: torch.Tensor) -> torch.Tensor:
    """A: (H, L, L) cuda float -> (H, 5) features."""
    Ld = A.shape[1]
    idx = torch.arange(Ld, device=A.device)
    f_key0 = A[:, :, 0].mean(dim=1)
    f_diag = A[:, idx, idx].mean(dim=1)
    f_next = A[:, idx, (idx + 1) % Ld].mean(dim=1)
    f_prev = A[:, idx, (idx - 1) % Ld].mean(dim=1)
    f_peak = A.max(dim=2).values.mean(dim=1)
    return torch.stack([f_key0, f_diag, f_next, f_prev, f_peak], dim=1)


# Prototypes: features of the five *clean* templates, in MODES order
# (used by the strawman; the full classifier uses a scale-invariant rule).
_templates_np = np.stack([_BUILD[m](L) for m in MODES], axis=0)            # (5, L, L)
TEMPLATES = torch.as_tensor(_templates_np, dtype=torch.float32, device=DEVICE)
PROTO = _features(TEMPLATES)                                                # (5, 5)

# Hand-set decision constants for the full (scale-invariant) classifier.
#   A head is mode X if band X carries more than TAU of the row-averaged mass;
#   if no band beats TAU the head is `uniform`. This argmax-vs-threshold rule
#   ignores the *absolute* spike height, so it survives noise that merely
#   shrinks every spike toward the uniform floor (1/L = 0.0625 here).
TAU = 0.18          # uniform-vs-spiked threshold on row-averaged band mass
BETA_FULL = 25.0    # softmax sharpness


def model_fn(attention_matrices: np.ndarray) -> np.ndarray:
    """Scale-invariant band-argmax classifier with a uniform threshold (GPU)."""
    A = torch.as_tensor(attention_matrices, dtype=torch.float32, device=DEVICE)
    feat = _features(A)                                                     # (H, 5)
    f_key0, f_diag, f_next, f_prev, _ = feat.unbind(dim=1)
    tau = torch.full_like(f_key0, TAU)
    # logits in MODES order: positional, uniform, diagonal, induction, previous
    logits = torch.stack([f_key0, tau, f_diag, f_next, f_prev], dim=1)      # (H, 5)
    probs = torch.softmax(BETA_FULL * logits, dim=1)
    return probs.detach().cpu().numpy().astype(np.float32)


def strawman_fn(attention_matrices: np.ndarray) -> np.ndarray:
    """Naive nearest-clean-prototype by absolute L2 distance (no scale
    invariance) -> collapses toward `uniform` as noise shrinks the spikes."""
    A = torch.as_tensor(attention_matrices, dtype=torch.float32, device=DEVICE)
    feat = _features(A)                                                     # (H, 5)
    dist2 = ((feat[:, None, :] - PROTO[None, :, :]) ** 2).sum(dim=2)        # (H, 5)
    probs = torch.softmax(-BETA * dist2, dim=1)
    return probs.detach().cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    run_dir = results_dir(__file__)

    # --- canonical evaluation (the real attempt) ---
    payload = task.evaluate(model_fn)

    # --- demo artefacts: per-head matrices, features, full & strawman preds ---
    all_mats, all_feat, all_full, all_straw, all_true, all_noise = [], [], [], [], [], []
    for noise in task.NOISE_LEVELS:
        batch = task.generate(task.CANONICAL_SEED, noise)
        mats = batch.attention_matrices                                    # (50, L, L)
        A = torch.as_tensor(mats, dtype=torch.float32, device=DEVICE)
        feat = _features(A).detach().cpu().numpy()
        full = model_fn(mats)
        straw = strawman_fn(mats)
        true_idx = np.array([MODES.index(m) for m in batch.true_modes], dtype=np.int64)
        all_mats.append(mats.astype(np.float32))
        all_feat.append(feat.astype(np.float32))
        all_full.append(full.astype(np.float32))
        all_straw.append(straw.astype(np.float32))
        all_true.append(true_idx)
        all_noise.append(np.full(len(true_idx), float(noise), dtype=np.float32))

    np.savez_compressed(
        run_dir / "demo.npz",
        matrices=np.concatenate(all_mats, axis=0),
        feats=np.concatenate(all_feat, axis=0),
        full_probs=np.concatenate(all_full, axis=0),
        straw_probs=np.concatenate(all_straw, axis=0),
        true_idx=np.concatenate(all_true, axis=0),
        noise=np.concatenate(all_noise, axis=0),
        prototypes=PROTO.detach().cpu().numpy().astype(np.float32),
        modes=np.array(MODES),
        L=np.int64(L),
    )

    record_benchmark(__file__, run_dir, payload)
    print(f"Benchmark + demo artefacts recorded to {run_dir}")
