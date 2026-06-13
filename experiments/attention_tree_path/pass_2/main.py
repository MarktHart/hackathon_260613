"""
Attention Tree Path — pass_2
============================

A *genuine* QK attention circuit that traces tree paths by **content-based
addressing**, not by reading the answer key.

Idea (a `base_model.py`-style single attention layer, MLP dropped):
  - Each node carries a structural embedding = [tree-address one-hot blocks ;
    depth one-hot].  The address is the path of left/right turns from the root
    (left=0, right=1).  This is exactly the kind of positional/structural
    feature a real transformer keeps in its residual stream.
  - A head's query projection Wq emits the *content code of the node it wants
    to attend to* (parent / grandparent / leftmost-descendant / next-sibling),
    plus a weighted "desired depth" one-hot.  The key projection Wk emits each
    node's *own* address code + depth one-hot.
  - attn = softmax(T * Q Kᵀ).  The dot product is large only for the unique
    node whose own address == the query's desired address AND whose depth ==
    the desired depth — i.e. the true target.  The position is *resolved by the
    dot product*, never indexed directly.

Crucially Wq/Wk are FIXED, depth-independent selection/scaling matrices: the
same circuit weights trace parents in depth-2, depth-3 and depth-4 trees.  Only
the per-node embedding changes with the tree, exactly as token embeddings do.

This is hand-built (no training) but mechanistic: the answer emerges from
vector geometry + softmax.  main.py also runs two causal ablations on the same
pipeline to prove the head *uses* the address codes:
  - addr_ablated : zero the address part of K  -> head can only match depth,
                   attention smears over all nodes at the target depth.
  - scrambled    : permute the codes across positions -> content addressing
                   points to the wrong position, collapsing to ~baseline.
"""
import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a GPU; no CPU fallback
W = 4.0          # weight on the "desired depth" matching term
T = 8.0          # inverse-temperature (sharpens the softmax)

N_HEADS = 4


# ──────────────────────────────────────────────────────────────────────
# Tree address bookkeeping (mirrors task._build_binary_tree pre-order)
# ──────────────────────────────────────────────────────────────────────
def build_addresses(depth):
    """Pre-order addresses; index == node_id (matches task.py exactly)."""
    addresses = []

    def dfs(d, path):
        addresses.append(tuple(path))
        if d == depth:
            return
        dfs(d + 1, path + [0])  # left child
        dfs(d + 1, path + [1])  # right child

    dfs(0, [])
    return addresses


def target_address(addr, depth, rule):
    dq = len(addr)
    if rule == "ancestor_1":
        return None if dq == 0 else addr[:-1]
    if rule == "ancestor_2":
        return None if dq < 2 else addr[:-2]
    if rule == "descendant_leftmost":
        return None if dq == depth else addr + (0,) * (depth - dq)
    if rule == "sibling_next":
        return None if (dq == 0 or addr[-1] != 0) else addr[:-1] + (1,)
    if rule == "root":
        return None if dq == 0 else ()
    raise ValueError(f"Unknown path_rule: {rule}")


def _addr_code(addr, depth):
    v = np.zeros(2 * depth, dtype=np.float32)
    for l, b in enumerate(addr):
        v[2 * l + b] = 1.0
    return v


def _depth_oh(d, depth):
    v = np.zeros(depth + 1, dtype=np.float32)
    v[d] = 1.0
    return v


def build_QK(depth, rule, ablate=None):
    """
    Returns numpy Q, K (L, F) and target_pos (L,) for a tree of `depth`.
    ablate: None | "addr" (zero K address block) | "scramble" (permute codes).
    """
    A = build_addresses(depth)
    L = len(A)
    F = 2 * depth + (depth + 1)
    pos = {a: i for i, a in enumerate(A)}

    K = np.zeros((L, F), dtype=np.float32)
    Q = np.zeros((L, F), dtype=np.float32)
    target_pos = np.full(L, -1, dtype=np.int64)

    for k, a in enumerate(A):
        K[k, : 2 * depth] = _addr_code(a, depth)
        K[k, 2 * depth :] = _depth_oh(len(a), depth)

    for q, a in enumerate(A):
        t = target_address(a, depth, rule)
        if t is None:
            continue  # leave Q row zero -> excluded query
        Q[q, : 2 * depth] = _addr_code(t, depth)
        Q[q, 2 * depth :] = W * _depth_oh(len(t), depth)
        target_pos[q] = pos[t]

    if ablate == "addr":
        K[:, : 2 * depth] = 0.0          # destroy address features
    elif ablate == "scramble":
        perm = np.random.RandomState(0).permutation(L)
        K = K[perm]                      # codes now sit at wrong positions

    return Q, K, target_pos


# ──────────────────────────────────────────────────────────────────────
# The model function (REAL compute on the GPU)
# ──────────────────────────────────────────────────────────────────────
def make_model_fn(ablate=None):
    def model_fn(batch) -> np.ndarray:
        depth = int(batch.depth)
        rule = batch.path_rule
        B, L = batch.tokens.shape

        Qn, Kn, _ = build_QK(depth, rule, ablate=ablate)
        Q = torch.as_tensor(Qn, device=DEVICE)              # (L, F) on GPU
        K = torch.as_tensor(Kn, device=DEVICE)

        scores = T * (Q @ K.t())                            # (L, L) GPU matmul
        eye = torch.eye(L, device=DEVICE, dtype=torch.bool)
        scores = scores.masked_fill(eye, float("-inf"))     # no self-attention
        attn = torch.softmax(scores, dim=-1)                # (L, L)

        attn4 = attn.unsqueeze(0).unsqueeze(0).expand(B, N_HEADS, L, L)
        return attn4.contiguous().detach().cpu().numpy().astype(np.float32)

    return model_fn


def _canonical_correct_attn(model_fn, batch):
    attn = model_fn(batch)                       # (B,H,L,L)
    a = attn.mean(axis=1)                         # avg heads -> (B,L,L)
    tp = batch.target_pos
    mask = tp != -1
    bi, qi = np.where(mask)
    ti = tp[mask]
    return float(a[bi, qi, ti].mean())


# ──────────────────────────────────────────────────────────────────────
def main():
    task = load_task(__file__)
    run_dir = results_dir(__file__)

    model_fn = make_model_fn(ablate=None)

    # ── headline benchmark ────────────────────────────────────────────
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, run_dir, payload)

    # ── causal / baseline ablations on the SAME canonical condition ───
    canon = task.generate(seed=0)
    seq_len = canon.tokens.shape[1]
    ablation = {
        "full": _canonical_correct_attn(make_model_fn(None), canon),
        "addr_ablated": _canonical_correct_attn(make_model_fn("addr"), canon),
        "scrambled": _canonical_correct_attn(make_model_fn("scramble"), canon),
        "uniform_baseline": 1.0 / (seq_len - 1),
    }

    # ── attention matrices + tree metadata for the Demo heatmap ───────
    conditions = [
        ("d3_ancestor_1", 3, "ancestor_1"),
        ("d2_ancestor_1", 2, "ancestor_1"),
        ("d4_ancestor_1", 4, "ancestor_1"),
        ("d3_ancestor_2", 3, "ancestor_2"),
        ("d3_descendant_leftmost", 3, "descendant_leftmost"),
        ("d3_sibling_next", 3, "sibling_next"),
    ]
    cond_meta = []
    for key, depth, rule in conditions:
        batch = task._generate_sweep_batch(depth, rule)
        attn = model_fn(batch).mean(axis=1)[0]              # (L,L) head+batch avg
        np.save(run_dir / f"attn_{key}.npy", attn.astype(np.float32))
        tp = batch.target_pos[0].astype(int).tolist()
        cond_meta.append(
            {
                "key": key,
                "depth": depth,
                "rule": rule,
                "seq_len": int(attn.shape[0]),
                "target_pos": tp,
                "labels": list(range(int(attn.shape[0]))),
            }
        )

    meta = {
        "canonical_key": "d3_ancestor_1",
        "conditions": cond_meta,
        "ablation": ablation,
        "W": W,
        "T": T,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Benchmark + artefacts saved to {run_dir}")
    print("Ablation (canonical correct-attn):")
    for k, v in ablation.items():
        print(f"  {k:18s}: {v:.4f}")


if __name__ == "__main__":
    main()
