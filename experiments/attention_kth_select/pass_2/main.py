"""
Per-sequence content->position pointer head for k-th position selection.

Attempt type: hand_built (no training).

WHY THIS DIFFERS FROM first_pass
--------------------------------
first_pass recovered k by averaging the marker mask across the *whole batch*
(pos_freq = marker_mask.mean(dim=0); k_hat = argmax). That is a cross-sequence
statistic no real attention head can compute -- a head processes each sequence
independently -- so its perfect attn_at_k=1.0 was an artefact of the eval's
batched structure, not a faithful mechanism. The jury flagged exactly this.

This attempt computes everything PER SEQUENCE. Within a single sequence the
position k is only revealed by the marker token (M=99) sitting there; but the
marker also lands at other positions by chance (~1/V each). So the honest,
real-head mechanism is a content pointer: form a fixed query from the marker
embedding and attend (positionally) to whichever positions carry it.

THE KEY INTERP RESULT
---------------------
Per sequence, every position holding the marker is *exchangeable* with the true
k (the posterior over "which marker is the forced one" is uniform). Hence the
Bayes-optimal expected mass on k is

    accuracy* = E[ 1 / (1 + S) ],   S ~ Binomial(L-1, r),   r = P(spurious marker)

With L=32, r=1/V=0.01 this is ~0.859 -- NOT 1.0. We show the hand-built pointer
hits this analytic ceiling, that 1.0 is only reachable with the (unfaithful)
cross-sequence oracle, and exactly where the mechanism degrades as r grows.

THE CIRCUIT (base_model.py reduced to one attention head, hand-set Q/K)
----------------------------------------------------------------------
  token embedding   E = I_V               (one-hot per token id)
  keys              K[b,l] = E[ ids[b,l] ]            (B, L, V)
  query (fixed)     q = beta * E[M]                   (V,)   -- hand-set
  scores            s[b,l] = <K[b,l], q> = beta if ids[b,l]==M else 0
  attn              softmax(s, dim=1)
No MLP, no value projection -- the metric scores the attention pattern only.
All compute runs in torch on CUDA.
"""

import json
import math

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU; no CPU fallback.

task = load_task(__file__)

V = 100
MARKER = 99
L = 32
BETA = 30.0  # inverse temperature of the (hand-set) marker query


# --------------------------------------------------------------------------- #
# Hand-built model functions (all real torch compute on CUDA)
# --------------------------------------------------------------------------- #
def content_pointer_fn(input_ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """FAITHFUL, per-sequence. Real QK attention head keyed on the marker.

    The query q = BETA * onehot(MARKER) is a single hand-set vector; keys are
    one-hot token embeddings. No information crosses sequences.
    """
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=DEVICE)        # (B, L)
    keys = torch.nn.functional.one_hot(ids, num_classes=V).float()           # (B, L, V)
    query = torch.zeros(V, device=DEVICE)
    query[MARKER] = BETA                                                      # hand-set
    scores = keys @ query                                                    # (B, L)
    attn = torch.softmax(scores, dim=1)
    return attn.detach().cpu().numpy().astype(np.float32)


def query_ablated_fn(input_ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """ABLATION: zero the query (beta=0). Scores collapse -> uniform attention."""
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=DEVICE)
    keys = torch.nn.functional.one_hot(ids, num_classes=V).float()
    query = torch.zeros(V, device=DEVICE)  # marker weight removed
    scores = keys @ query
    attn = torch.softmax(scores, dim=1)
    return attn.detach().cpu().numpy().astype(np.float32)


def marker_channel_ablated_fn(input_ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """ABLATION: keep the query but zero the marker channel of every key.

    The head can no longer read the dimension that distinguishes the marker, so
    every position looks identical -> uniform. Causal check that the head's
    behaviour is driven specifically by the marker channel.
    """
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=DEVICE)
    keys = torch.nn.functional.one_hot(ids, num_classes=V).float()
    keys[..., MARKER] = 0.0                                                   # ablate channel
    query = torch.zeros(V, device=DEVICE)
    query[MARKER] = BETA
    scores = keys @ query
    attn = torch.softmax(scores, dim=1)
    return attn.detach().cpu().numpy().astype(np.float32)


def make_fixed_position_fn(p0: int):
    """STRAWMAN: a pure positional head hard-wired to a FIXED position p0.

    Attends to p0 regardless of content -- nails k=p0 but fails for every other
    k, because it has no way to track the varying target. This is what
    'position addressing without knowing k' looks like.
    """
    def _fn(input_ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=DEVICE)
        B, Ll = ids.shape
        pos = torch.as_tensor(positions, dtype=torch.long, device=DEVICE)     # (L,)
        scores = BETA * (pos == p0).float()                                   # (L,)
        attn = torch.softmax(scores, dim=0).unsqueeze(0).expand(B, -1).contiguous()
        return attn.detach().cpu().numpy().astype(np.float32)
    return _fn


def oracle_batch_fn(input_ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """UNFAITHFUL UPPER BOUND (first_pass's mechanism, kept only for contrast).

    Aggregates the marker mask across the batch to read k exactly, then emits a
    positional delta. A real per-sequence head CANNOT do this. Labelled as an
    oracle so the gap to the faithful pointer is visible.
    """
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=DEVICE)
    B, Ll = ids.shape
    pos_freq = (ids == MARKER).float().mean(dim=0)                            # (L,) cross-seq
    k_hat = torch.argmax(pos_freq)
    scores = BETA * (torch.arange(Ll, device=DEVICE) == k_hat).float()
    attn = torch.softmax(scores, dim=0).unsqueeze(0).expand(B, -1).contiguous()
    return attn.detach().cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# Stats helpers (numpy, for the visualisation artefacts)
# --------------------------------------------------------------------------- #
def _entropy(attn: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.mean(np.sum(attn * np.log(attn + eps), axis=1)))


def _method_records(fn) -> list[dict]:
    log_L = math.log(L)
    out = []
    for batch in task.generate(seed=0):
        attn = fn(batch.input_ids, batch.positions)                          # (B, L)
        k = int(batch.target_k)
        ent = _entropy(attn)
        out.append({
            "k": k,
            "attn_at_k": float(attn[:, k].mean()),
            "sharpness": float(max(0.0, min(1.0, 1.0 - ent / log_L))),
            "attn_max_pos": float(np.mean(np.argmax(attn, axis=1))),
            "mean_attn": attn.mean(axis=0).astype(float).tolist(),
        })
    return out


def _analytic_accuracy(r: float, length: int = L) -> float:
    """Bayes-optimal E[1/(1+S)], S ~ Binomial(length-1, r)."""
    n = length - 1
    acc = 0.0
    for s in range(n + 1):
        p = math.comb(n, s) * (r ** s) * ((1.0 - r) ** (n - s))
        acc += p / (1.0 + s)
    return float(acc)


def _spurious_batch(r: float, B: int, k: int, seed: int) -> np.ndarray:
    """Sequence batch where each non-k position is a marker w.p. r (exactly).

    Noise drawn from 0..V-2 (never the marker) so the spurious rate is exactly r.
    """
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, V - 1, size=(B, L), dtype=np.int32)                 # 0..98, no marker
    extra = rng.random((B, L)) < r
    ids[extra] = MARKER
    ids[:, k] = MARKER                                                        # force k
    return ids


# --------------------------------------------------------------------------- #
def main() -> None:
    run_dir = results_dir(__file__)
    positions = np.arange(L, dtype=np.int32)

    # ---- Official benchmark payload: the FAITHFUL per-sequence pointer ----
    payload = task.evaluate(content_pointer_fn)
    payload["model_name"] = "handbuilt-content-pointer-persequence(bayes-optimal)"

    # ---- Sweep comparison across methods ----
    methods = {
        "content_pointer": _method_records(content_pointer_fn),
        "fixed_position@8": _method_records(make_fixed_position_fn(8)),
        "oracle_batch": _method_records(oracle_batch_fn),
        "uniform": _method_records(task.random_model_fn()),
    }

    # ---- Operating-range sweep over spurious-marker rate r ----
    operating_range = []
    for r in [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50]:
        ids = _spurious_batch(r, B=512, k=8, seed=123)
        attn = content_pointer_fn(ids, positions)
        operating_range.append({
            "r": float(r),
            "empirical_acc": float(attn[:, 8].mean()),
            "analytic_acc": _analytic_accuracy(r),
        })

    # ---- Ablation at canonical k=8 ----
    canon = next(b for b in task.generate(seed=0) if b.target_k == 8)
    ablation = {
        "content_pointer": float(content_pointer_fn(canon.input_ids, positions)[:, 8].mean()),
        "query_ablated": float(query_ablated_fn(canon.input_ids, positions)[:, 8].mean()),
        "marker_channel_ablated": float(
            marker_channel_ablated_fn(canon.input_ids, positions)[:, 8].mean()
        ),
        "uniform_baseline": 1.0 / L,
    }

    comparison = {
        "L": L,
        "V": V,
        "marker": MARKER,
        "beta": BETA,
        "sweep_k": [int(b.target_k) for b in task.generate(seed=0)],
        "analytic_ceiling": _analytic_accuracy(1.0 / V),  # ~0.859 for the task's r
        "uniform_baseline": 1.0 / L,
        "methods": methods,
        "operating_range": operating_range,
        "ablation": ablation,
    }
    with open(run_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    record_benchmark(__file__, run_dir, payload)
    print(f"wrote benchmark + comparison to {run_dir}")
    print("canonical attn_at_k (content pointer):",
          round(payload["sweep"][2]["attn_at_k"], 4),
          "| analytic ceiling:", round(comparison["analytic_ceiling"], 4))
    print("ablation:", {k: round(v, 4) for k, v in ablation.items()})


if __name__ == "__main__":
    main()
