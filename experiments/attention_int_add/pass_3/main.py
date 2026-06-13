"""
attention_int_add / pass_3  —  carry propagation AS ATTENTION (carry-lookahead).

Why this attempt exists
-----------------------
pass_2 was a clean hand-built circuit that aced every slice, but the jury's main
complaint was *architecture fit*: its carry propagation was an explicit Python
`for`-loop ripple (`carry = (total >= 10)` iterated column by column). The goal
asks attention to perform the inter-position routing of carries; a sequential
Python loop is exactly NOT that.

This attempt fixes the mechanism. Carry propagation here is a genuine **softmax
attention layer** — a carry-lookahead adder — with NO sequential loop over
columns. The whole carry chain is resolved in one parallel attention op:

  For each answer column i, the query attends over the lower columns j < i and
  lands (one-hot, via a hand-set score) on the *nearest decisive* column: the
  most-significant column below i whose digit-sum is NOT 9. Columns whose sum is
  exactly 9 are "propagators" — they pass a carry straight through — so the score
  gives them a large negative bias, and attention skips over the entire run of
  9s to read the generate/kill signal underneath. The value it reads is
  g_j = [s_j >= 10] (does column j generate a carry). That single attention read
  IS the carry into column i, for every column at once.

This is the textbook carry-lookahead identity expressed as attention:
    carry_in(i) = g_{j*}   where  j* = max{ j < i : s_j != 9 }
because every column strictly between j* and i has s==9 and therefore forwards
whatever carry reaches it. 999+1 → the THOUSANDS column attends two positions
down, past both 9s, all the way to the units column. That long-range hop is the
carry chain, and it is attention.

Model = `base_model.py` delta
-----------------------------
- token embedding  -> a hand-set *value* embedding (id -> digit value).
- attention layer 1 (digit routing): hand-set near-one-hot heads pull operand
  digits a_j, b_j into per-column slots. (the "easy part" — column fetch)
- a small MLP-style nonlinearity: s_j = a_j+b_j, g_j=[s_j>=10], p_j=[s_j==9].
- attention layer 2 (carry-lookahead): the real contribution above. One softmax
  attention over columns resolves the full carry chain in parallel.
- unembed: digit_i = (s_i + carry_i) mod 10, leading digit = top carry.

Everything is hand-set torch on CUDA (GPU guard). Nothing is trained, so the
mechanism is known by construction and we can ablate the carry-attention layer
exactly. The carry-lookahead is digit-width agnostic, so we also demonstrate it
holds across 3..12 digit operands (operands up to 10^12) — carry chains 4x the
canonical length — to show the mechanism is the adder, not a 3-digit lookup.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # GPU guaranteed; no CPU fallback.

# ---- constants mirroring task.py -----------------------------------------
VOCAB_SIZE = 15
SEQ_LEN = 14
MAX_DIGITS = 3
SUM_DIGITS = 4
SUM_POSITIONS = [9, 10, 11, 12]     # MSB-first: thousands, hundreds, tens, units

# Operand digit source positions (sequence is MSB-first):
#   a: idx1 hundreds, idx2 tens, idx3 units ;  b: idx5,6,7
# Column index is LSB-first: c=0 units, c=1 tens, c=2 hundreds.
A_SRC = {0: 3, 1: 2, 2: 1}          # column -> a source index
B_SRC = {0: 7, 1: 6, 2: 5}          # column -> b source index

# carry-lookahead score constants (large gaps -> softmax ~ one-hot / argmax)
POS_SCALE = 10.0                    # prefer the most-significant valid column
MASK_BIG = 1.0e4                    # forbid attending to self / higher columns
PROP_BIG = 1.0e3                    # skip "propagator" columns (digit-sum == 9)


# ==========================================================================
# Carry-lookahead carry as a SINGLE softmax-attention op (no column loop)
# ==========================================================================
def carry_lookahead_attention(s: torch.Tensor):
    """
    s : (N, D) float tensor of per-column digit sums (a_j + b_j), LSB-first.

    Returns
    -------
    carry   : (N, D+1) carry INTO each column (carry[:,0]=0); carry[:,D] is the
              overflow / leading digit. Each entry is 0 or 1.
    weights : (N, D+1, D) the attention pattern (query=target column incl.
              overflow, key=source column). Saved for visualisation.

    Mechanism: query i attends over keys j with score
        score(i,j) = POS_SCALE*j  - MASK_BIG*[j >= i]  - PROP_BIG*[s_j == 9]
    softmax -> ~one-hot on the most-significant j < i with s_j != 9. The value
    read is g_j = [s_j >= 10]. No iteration over columns: the whole (D+1, D)
    score tensor is built and softmaxed at once, so a length-D carry chain is
    resolved in one parallel attention.
    """
    N, D = s.shape
    g = (s >= 10.0).float()                      # generate signal (value read)
    p = (s == 9.0).float()                       # propagator mask

    j_idx = torch.arange(D, device=s.device, dtype=s.dtype)         # keys
    i_idx = torch.arange(D + 1, device=s.device, dtype=s.dtype)     # queries

    base = POS_SCALE * j_idx.view(1, 1, D)                          # (1,1,D)
    causal = (j_idx.view(1, 1, D) >= i_idx.view(1, D + 1, 1)).float() * (-MASK_BIG)
    prop = p.view(N, 1, D) * (-PROP_BIG)                            # (N,1,D)

    scores = base + causal + prop                                  # (N, D+1, D)
    weights = torch.softmax(scores, dim=-1)                        # ~one-hot
    carry = (weights * g.view(N, 1, D)).sum(dim=-1)                # (N, D+1)
    carry[:, 0] = 0.0                                              # col 0 has no input
    return carry.round(), weights


# ==========================================================================
# Hand-built circuit over the task's sequence layout
# ==========================================================================
class CarryLookaheadAdder:
    """Hand-set torch circuit. `ablate_carry=True` removes the carry-attention
    layer (carries forced to 0) -> collapses to the task's linear baseline."""

    def __init__(self, ablate_carry: bool = False):
        self.ablate_carry = ablate_carry

        # value embedding: token id -> digit value, specials -> 0
        val = torch.zeros(VOCAB_SIZE, device=DEVICE)
        for t in range(10):
            val[t] = float(t)
        self.val_emb = val

        # digit-routing attention: near-one-hot heads (softmax of hard scores)
        sa = torch.full((MAX_DIGITS, SEQ_LEN), -1e9, device=DEVICE)
        sb = torch.full((MAX_DIGITS, SEQ_LEN), -1e9, device=DEVICE)
        for c in range(MAX_DIGITS):
            sa[c, A_SRC[c]] = 0.0
            sb[c, B_SRC[c]] = 0.0
        self.attn_a = torch.softmax(sa, dim=-1)        # (3, SEQ_LEN)
        self.attn_b = torch.softmax(sb, dim=-1)

    @torch.no_grad()
    def run(self, input_ids: torch.Tensor):
        N = input_ids.shape[0]
        val = self.val_emb[input_ids]                  # (N, SEQ_LEN)

        # layer 1: route operand digits into per-column slots (attention)
        col_a = val @ self.attn_a.t()                  # (N, 3) LSB-first
        col_b = val @ self.attn_b.t()                  # (N, 3)
        s = col_a + col_b                              # column sums (N, 3)

        # layer 2: carry-lookahead attention (or ablated)
        if self.ablate_carry:
            carry = torch.zeros(N, MAX_DIGITS + 1, device=DEVICE)
            weights = torch.zeros(N, MAX_DIGITS + 1, MAX_DIGITS, device=DEVICE)
        else:
            carry, weights = carry_lookahead_attention(s)

        # unembed: digit_i = (s_i + carry_in_i) mod 10 ; lead = top carry
        digits = torch.remainder(s + carry[:, :MAX_DIGITS], 10.0)   # (N,3) LSB
        lead = carry[:, MAX_DIGITS]                                  # (N,)

        # MSB-first predicted digits: thousands, hundreds, tens, units
        pred = torch.stack([lead, digits[:, 2], digits[:, 1], digits[:, 0]], dim=1)
        pred = pred.round().long().clamp_(0, 9)                     # (N,4)

        logits = torch.zeros(N, SEQ_LEN, VOCAB_SIZE, device=DEVICE)
        ar = torch.arange(N, device=DEVICE)
        for i, pos in enumerate(SUM_POSITIONS):
            logits[ar, pos, pred[:, i]] = 30.0
        return logits, weights, s

    def model_fn(self, input_ids: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(input_ids, dtype=torch.long, device=DEVICE)
        logits, _, _ = self.run(x)
        return logits.detach().cpu().numpy()


# ==========================================================================
# Generalised D-digit adder (same carry-lookahead attention) for operating range
# ==========================================================================
@torch.no_grad()
def general_add_digits(a: np.ndarray, b: np.ndarray, D: int) -> np.ndarray:
    """Add a+b (each in [0,10^D)) using the carry-lookahead ATTENTION over D
    columns, on GPU. Returns predicted digit array (N, D+1), MSB-first."""
    at = torch.as_tensor(a, dtype=torch.long, device=DEVICE)
    bt = torch.as_tensor(b, dtype=torch.long, device=DEVICE)
    cols_a, cols_b = [], []
    for c in range(D):                                  # LSB-first columns
        powc = 10 ** c
        cols_a.append((at // powc) % 10)
        cols_b.append((bt // powc) % 10)
    s = (torch.stack(cols_a, 1) + torch.stack(cols_b, 1)).float()   # (N,D)
    carry, _ = carry_lookahead_attention(s)
    digits = torch.remainder(s + carry[:, :D], 10.0)                # (N,D) LSB
    lead = carry[:, D].view(-1, 1)                                  # (N,1)
    full = torch.cat([digits, lead], dim=1).round().long()         # (N,D+1) LSB..MSB
    msb = torch.flip(full, dims=[1])                                # MSB-first
    return msb.cpu().numpy()


def _true_digits(vals: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((vals.shape[0], n), dtype=np.int64)
    for i in range(n):
        out[:, i] = (vals // (10 ** (n - 1 - i))) % 10
    return out


# ==========================================================================
# Analysis helpers
# ==========================================================================
def predict_digits(model_fn, input_ids):
    logits = model_fn(input_ids)
    return np.argmax(logits[:, SUM_POSITIONS, :], axis=-1)


def per_carry_em(pred, batch):
    out = {}
    for k, (sidx, eidx) in zip(batch.carry_sweep, batch.slice_indices):
        if eidx > sidx:
            p = pred[sidx:eidx]
            t = batch.target_sum_digits[sidx:eidx]
            out[int(k)] = float((p == t).all(axis=1).mean())
        else:
            out[int(k)] = 0.0
    return out


def main():
    task = load_task(__file__)
    run_dir = results_dir(__file__)
    run_dir.mkdir(parents=True, exist_ok=True)

    full = CarryLookaheadAdder(ablate_carry=False)
    abl = CarryLookaheadAdder(ablate_carry=True)

    # ---- operating range over carry slices x held-out seeds -----------------
    seeds = list(range(8))
    sweep = None
    full_ps, abl_ps = {}, {}
    for sd in seeds:
        batch = task.generate(seed=sd)
        sweep = [int(k) for k in batch.carry_sweep]
        full_ps[sd] = per_carry_em(predict_digits(full.model_fn, batch.input_ids), batch)
        abl_ps[sd] = per_carry_em(predict_digits(abl.model_fn, batch.input_ids), batch)

    def mean_over(ps):
        return {int(k): float(np.mean([ps[sd][int(k)] for sd in seeds])) for k in sweep}

    full_mean, abl_mean = mean_over(full_ps), mean_over(abl_ps)

    def robustness(em):
        easy = em.get(sweep[0], 0.0)
        hard = em.get(sweep[-1], 0.0)
        return float(max(0.0, min(1.0, hard / easy))) if easy > 1e-12 else 0.0

    faithfulness = {
        "seeds": seeds,
        "carry_sweep": sweep,
        "full_em_mean": {str(k): full_mean[k] for k in sweep},
        "ablated_em_mean": {str(k): abl_mean[k] for k in sweep},
        "carry_robustness_full": robustness(full_mean),
        "carry_robustness_ablated": robustness(abl_mean),
        "note": ("Ablating the carry-lookahead ATTENTION layer (carries forced to 0) "
                 "reduces the circuit to the task's linear no-carry baseline; "
                 "exact-match collapses on every carrying slice."),
    }
    with open(run_dir / "faithfulness.json", "w") as f:
        json.dump(faithfulness, f, indent=2)

    # ---- attention patterns for canonical examples (the key artefact) -------
    examples = [(99, 901), (999, 1), (456, 544), (123, 456), (500, 500), (1, 999)]
    a_arr = np.array([e[0] for e in examples], dtype=np.int64)
    b_arr = np.array([e[1] for e in examples], dtype=np.int64)
    s_cols = (_true_digits(a_arr, 3)[:, ::-1] + _true_digits(b_arr, 3)[:, ::-1])  # LSB
    st = torch.as_tensor(s_cols, dtype=torch.float32, device=DEVICE)
    carry, weights = carry_lookahead_attention(st)
    attn_records = []
    for i, (a, b) in enumerate(examples):
        attn_records.append({
            "a": int(a), "b": int(b), "true": int(a + b),
            "col_sums_lsb": [int(x) for x in s_cols[i]],
            "carry_in": [float(x) for x in carry[i].cpu().numpy()],
            "weights": weights[i].cpu().numpy().tolist(),  # (4 queries, 3 keys)
        })
    with open(run_dir / "attention.json", "w") as f:
        json.dump(attn_records, f, indent=2)

    # ---- operating range over DIGIT WIDTH (carry-chain length) --------------
    rng = np.random.default_rng(0)
    gen_records = []
    for D in range(3, 13):
        hi = 10 ** D
        a = rng.integers(0, hi, size=4000, dtype=np.int64)
        b = rng.integers(0, hi, size=4000, dtype=np.int64)
        # inject worst-case full carry chains: 10^D-1 + 1, plus random "all-9" sums
        a = np.concatenate([a, np.full(50, hi - 1), rng.integers(0, hi, 50)])
        b = np.concatenate([b, np.full(50, 1), (hi - 1) - rng.integers(0, hi, 50)])
        pred = general_add_digits(a, b, D)              # (N, D+1) MSB
        true = _true_digits(a + b, D + 1)
        em = float((pred == true).all(axis=1).mean())
        # linear (no-carry) baseline for the same width
        ad = _true_digits(a, D)[:, ::-1]
        bd = _true_digits(b, D)[:, ::-1]
        base_lsb = (ad + bd) % 10
        base = np.concatenate([base_lsb[:, ::-1], np.zeros((a.shape[0], 1), int)], axis=1)
        base_em = float((base == true).all(axis=1).mean())
        gen_records.append({"digits": D, "exact_match": em, "baseline_exact_match": base_em})
        print(f"  D={D:2d} digits  carry-lookahead EM={em:.3f}   linear baseline EM={base_em:.3f}")
    with open(run_dir / "generalization.json", "w") as f:
        json.dump(gen_records, f, indent=2)

    print("\nOperating range (mean exact-match over 8 seeds):")
    for k in sweep:
        print(f"  carries={k}:  full={full_mean[k]:.3f}   carry-ablated={abl_mean[k]:.3f}")

    # ---- canonical benchmark (seed=0, full circuit) -------------------------
    payload = task.evaluate(full.model_fn)
    record_benchmark(__file__, run_dir, payload)
    print(f"\nRecorded benchmark -> {run_dir}")


if __name__ == "__main__":
    main()
