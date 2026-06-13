"""pass_2 — Game of Life as a single self-attention layer + MLP (hand-built).

This is the SMALLEST useful delta from `base_model.py` (token-embed -> one
self-attention block -> MLP -> residual -> unembed) that solves Conway's Game
of Life. Unlike a conv2d, the neighbour-counting here is done by ATTENTION:

  * Each cell is a token (sequence length N = H*W).
  * The attention layer uses a hand-set *relative-position bias* (a toroidal
    neighbour mask) so every query cell attends uniformly (weight 1/8) to its
    eight neighbours and to nothing else. With V reading the cell's alive bit,
    the attention output is `mean(neighbour alive) = neighbour_count / 8`.
  * The output projection rescales that to the integer neighbour count `n`,
    which the residual stream carries alongside the cell's own state `s`.
  * A tiny hand-set ReLU MLP applies the GoL rule, written as
        alive_next  =  (n == 3)  OR  (n + s == 3)
    (two triangular "exactly-k" detectors), producing the per-cell logit.

Every weight is set by hand — no training — which both earns the hardcoded
bonus and makes the mechanism fully transparent. Faithfulness is checked with
a causal ablation: replacing the neighbour mask with a *global* uniform mask
(so attention can no longer localise to the 8 neighbours) collapses F1 to the
trivial regime, and zeroing the attention output ("self-only") drops it to 0.
We also sweep grid sizes 8..64 to show the same hand-set circuit holds across
~2 orders of magnitude of cell count.
"""

import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

# The pipeline guarantees a visible GPU; never fall back to CPU here.
DEVICE = "cuda"

D_MODEL = 2          # channel 0 = cell state s, channel 1 = neighbour count n
SCALE = 20.0         # logit sharpness


def _neighbour_bias(h: int, w: int) -> np.ndarray:
    """Additive attention bias: 0 for the 8 toroidal neighbours, -1e9 else."""
    n = h * w
    bias = np.full((n, n), -1e9, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            p = i * w + j
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    q = ((i + di) % h) * w + ((j + dj) % w)
                    bias[p, q] = 0.0
    return bias


def ref_next_state(grids: np.ndarray) -> np.ndarray:
    """Independent NumPy Conway GoL (toroidal) — ground truth for the demo."""
    grids = (np.asarray(grids) > 0.5).astype(np.int32)
    _, h, w = grids.shape
    padded = np.pad(grids, ((0, 0), (1, 1), (1, 1)), mode="wrap")
    counts = np.zeros_like(grids)
    for di in range(3):
        for dj in range(3):
            if di == 1 and dj == 1:
                continue
            counts += padded[:, di:di + h, dj:dj + w]
    alive = grids == 1
    survive = alive & ((counts == 2) | (counts == 3))
    birth = (~alive) & (counts == 3)
    return (survive | birth).astype(np.int32)


class GoLAttention:
    """base_model.py-shaped: embed -> 1 attention block -> MLP -> unembed.

    All weights are hand-set. `mode` selects causal interventions:
      * "full"     — neighbour mask (the real mechanism)
      * "ablate"   — global uniform attention (neighbour localisation removed)
      * "selfonly" — attention output zeroed (cell can't see any neighbour)
    """

    def __init__(self, h: int = 16, w: int = 16, device: str = DEVICE):
        self.h, self.w, self.n, self.device = h, w, h * w, device
        d = D_MODEL

        # --- attention projections (hand-set) ---
        # Q,K -> 0 so every base score is equal; the bias does the routing.
        self.Wq = torch.zeros(d, d, device=device)
        self.Wk = torch.zeros(d, d, device=device)
        # V reads channel 0 (the cell's alive bit) into a scalar.
        self.Wv = torch.tensor([[1.0], [0.0]], device=device)
        # O writes 8 * (mean neighbour alive) = neighbour count into channel 1.
        self.Wo = torch.tensor([[0.0, 8.0]], device=device)

        self.attn_bias = torch.tensor(_neighbour_bias(h, w), device=device)
        self.global_bias = torch.zeros(self.n, self.n, device=device)

        # --- MLP that applies the GoL rule on [s, n] ---
        # alive_next = (n == 3) OR (n + s == 3); each "== 3" is a ReLU triangle.
        # Hidden cols: relu(n-2),relu(n-3),relu(n-4), relu(m-2),relu(m-3),relu(m-4)
        # with m = n + s.
        W1 = np.array([[0, 0, 0, 1, 1, 1],    # weight on s
                       [1, 1, 1, 1, 1, 1]],   # weight on n
                      dtype=np.float32)
        b1 = np.array([-2, -3, -4, -2, -3, -4], dtype=np.float32)
        # T1 = col0 - 2*col1 + col2 (triangle peaking at n==3)
        # T2 = col3 - 2*col4 + col5 (triangle peaking at n+s==3)
        # logit = SCALE * (T1 + T2 - 0.5)  -> >0 iff at least one detector fires.
        W2 = np.array([[1], [-2], [1], [1], [-2], [1]], dtype=np.float32) * SCALE
        b2 = np.array([-0.5], dtype=np.float32) * SCALE
        self.W1 = torch.tensor(W1, device=device)
        self.b1 = torch.tensor(b1, device=device)
        self.W2 = torch.tensor(W2, device=device)
        self.b2 = torch.tensor(b2, device=device)

    @torch.no_grad()
    def forward(self, grid: torch.Tensor, mode: str = "full") -> torch.Tensor:
        """grid: (B, H, W) float on device -> logits (B, H, W)."""
        bn = grid.shape[0]
        state = grid.reshape(bn, self.n)                      # (B, N)
        x = torch.zeros(bn, self.n, D_MODEL, device=self.device)
        x[..., 0] = state                                     # token embedding

        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv                                       # (B, N, 1)
        scores = (q @ k.transpose(1, 2)) / (D_MODEL ** 0.5)   # (B, N, N) = 0
        bias = self.global_bias if mode == "ablate" else self.attn_bias
        attn = torch.softmax(scores + bias, dim=-1)
        o = attn @ v                                          # mean neighbour alive

        Wo = torch.zeros(1, D_MODEL, device=self.device) if mode == "selfonly" else self.Wo
        h = x + o @ Wo                                        # residual: ch0=s, ch1=n

        feats = torch.stack([h[..., 0], h[..., 1]], dim=-1)   # (B, N, 2)
        hidden = torch.relu(feats @ self.W1 + self.b1)
        logit = hidden @ self.W2 + self.b2                    # (B, N, 1)
        return logit[..., 0].reshape(bn, self.h, self.w)

    def model_fn(self, mode: str = "full"):
        def fn(grids: np.ndarray) -> np.ndarray:
            g = torch.as_tensor(grids, dtype=torch.float32, device=self.device)
            return self.forward(g, mode=mode).detach().cpu().numpy().astype(np.float32)
        return fn

    @torch.no_grad()
    def attention_row(self, query_idx: int, mode: str = "full") -> np.ndarray:
        """Attention weights from one query cell over all cells (for the demo)."""
        bias = self.global_bias if mode == "ablate" else self.attn_bias
        row = torch.softmax(bias[query_idx], dim=-1)
        return row.detach().cpu().numpy().reshape(self.h, self.w)


# ---- small metric helpers (for the ablation / robustness artefacts) ----
def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else 0.0


def _per_density_f1(payload: dict, prefix: str = "") -> list:
    return [_f1(r[f"{prefix}tp"], r[f"{prefix}fp"], r[f"{prefix}fn"])
            for r in payload["sweep"]]


def _f1_np(pred_live: np.ndarray, true_live: np.ndarray) -> float:
    p = pred_live.astype(bool)
    t = true_live.astype(bool)
    tp = int(np.sum(p & t)); fp = int(np.sum(p & ~t)); fn = int(np.sum(~p & t))
    return _f1(tp, fp, fn)


def robustness_sweep(device: str = DEVICE) -> list:
    """Same hand-set circuit, rebuilt across grid sizes 8..64."""
    out = []
    for s in (8, 16, 32, 64):
        model = GoLAttention(h=s, w=s, device=device)
        for d in (0.1, 0.3, 0.5):
            rng = np.random.default_rng([12345, s, int(d * 100)])
            g = (rng.random((4, s, s)) < d).astype(np.float32)
            true = ref_next_state(g)
            logit = model.forward(
                torch.as_tensor(g, dtype=torch.float32, device=device)
            ).detach().cpu().numpy()
            out.append({"size": s, "density": float(d), "f1": _f1_np(logit > 0, true)})
    return out


if __name__ == "__main__":
    task = load_task(__file__)
    model = GoLAttention(device=DEVICE)

    # Headline: the real attention circuit.
    payload = task.evaluate(model.model_fn("full"))
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    # Causal ablations (stored as a side artefact, not the headline).
    abl = task.evaluate(model.model_fn("ablate"))
    self_only = task.evaluate(model.model_fn("selfonly"))
    comparison = {
        "densities": payload["density_sweep"],
        "full_f1": _per_density_f1(payload),
        "ablate_f1": _per_density_f1(abl),
        "selfonly_f1": _per_density_f1(self_only),
        "static_f1": _per_density_f1(payload, prefix="static_"),
    }
    (run_dir / "ablation.json").write_text(json.dumps(comparison, indent=2))

    # Operating range across grid sizes.
    (run_dir / "robustness.json").write_text(
        json.dumps(robustness_sweep(DEVICE), indent=2)
    )

    print(f"Benchmark + ablation + robustness written to {run_dir}")
