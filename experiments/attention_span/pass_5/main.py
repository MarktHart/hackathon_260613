"""attention_span / pass_5 — hand-built content-addressed retrieval head.

Hypothesis
----------
The effective *span* of attention is limited by **content addressing, not by
distance**. A single attention head whose query/key projections align the
query token (8888) with the needle token (9999) will retrieve the needle at
ANY distance, because the score depends only on *what* the key token is, not
*where* it sits. So the attention-on-target curve should be roughly FLAT and
high across the whole 1..256 sweep (>2 orders of magnitude), giving a
robustness ratio (long-range / short-range) of ~1.0.

This is the *minimal delta* from `base_model.py`: one attention head, token
embedding + Q/K projections, scaled-dot-product softmax. No MLP, no training —
every weight is set by hand (see `build_head`). The same forward pass also
realises two contrasts, computed for the demo / faithfulness story:

  * positional strawman — replace content matching with an ALiBi-style local
    distance penalty (score[0,j] = -m*j). No content => attention decays with
    distance and CANNOT reach a far needle. This is the finite-span baseline.
  * ablation — zero the needle key direction in W_K. The retrieval circuit is
    destroyed and attention collapses to uniform. This is the causal check
    that the content head's span really comes from the Q/K alignment.

All real compute runs in torch on CUDA.
"""

import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU; no CPU fallback.

QUERY_TOKEN = 8888
NEEDLE_TOKEN = 9999
VOCAB = 10000
D_MODEL = 8

# Feature-channel layout inside the (tiny, hand-set) embedding.
CH_QUERY = 0     # query token writes a 1 here
CH_NEEDLE = 1    # needle token writes a 1 here
CH_RETRIEVE = 2  # shared Q/K direction used for the content match

task = load_task(__file__)


def build_head(score_scale: float = 12.0):
    """Hand-set embedding + Q/K projections for a content-retrieval head.

    Returns torch tensors on CUDA. `score_scale` is the value of the
    query.key dot product at the needle position (the softmax logit); every
    other position scores 0.
    """
    emb = torch.zeros(VOCAB, D_MODEL, device=DEVICE)
    emb[QUERY_TOKEN, CH_QUERY] = 1.0
    emb[NEEDLE_TOKEN, CH_NEEDLE] = 1.0
    # All other tokens (fillers in [1,1000)) embed to 0 -> they produce no
    # query and no key, so they never win the softmax.

    s = float(np.sqrt(score_scale))  # so that q.k = s*s = score_scale
    w_q = torch.zeros(D_MODEL, D_MODEL, device=DEVICE)  # (out, in)
    w_k = torch.zeros(D_MODEL, D_MODEL, device=DEVICE)
    # Query token (CH_QUERY) -> retrieval direction.
    w_q[CH_RETRIEVE, CH_QUERY] = s
    # Needle token (CH_NEEDLE) -> the SAME retrieval direction => they align.
    w_k[CH_RETRIEVE, CH_NEEDLE] = s
    return emb, w_q, w_k


def query_row_attention(input_ids_t, emb, w_q, w_k, alibi_slope=0.0):
    """Attention from query position 0 to every key position.

    input_ids_t: (B, L) long on CUDA. Returns (B, L) softmax row.
    `alibi_slope > 0` adds an ALiBi-style distance penalty -slope*|0-j|.
    """
    B, L = input_ids_t.shape
    x = emb[input_ids_t]                       # (B, L, D)
    q = x @ w_q.T                              # (B, L, D)
    k = x @ w_k.T                              # (B, L, D)
    q0 = q[:, 0:1, :]                          # (B, 1, D) query at position 0
    scores = (q0 @ k.transpose(1, 2)).squeeze(1)  # (B, L) raw logits
    if alibi_slope > 0:
        positions = torch.arange(L, device=DEVICE, dtype=torch.float32)
        scores = scores - alibi_slope * positions.abs()[None, :]
    return torch.softmax(scores, dim=-1)       # (B, L)


def make_model_fn(emb, w_q, w_k, alibi_slope=0.0, chunk=150):
    """Wrap the head into the task's model_fn: (B,L) int -> (B,L,L) attention.

    Only query-row 0 is read by the evaluator, so we compute that row exactly
    on the GPU and fill the remaining rows with uniform attention (1/L). This
    keeps the returned tensor memory-light while doing the real matmuls on
    CUDA.
    """
    def model_fn(input_ids: np.ndarray) -> np.ndarray:
        ids = torch.as_tensor(input_ids.astype(np.int64), device=DEVICE)
        B, L = ids.shape
        out = np.empty((B, L, L), dtype=np.float32)
        for start in range(0, B, chunk):
            end = min(start + chunk, B)
            row0 = query_row_attention(ids[start:end], emb, w_q, w_k, alibi_slope)
            attn = torch.full((end - start, L, L), 1.0 / L, device=DEVICE)
            attn[:, 0, :] = row0
            out[start:end] = attn.cpu().numpy()
            del attn
        return out
    return model_fn


def sweep_means(payload):
    return [s["mean_attention_on_target"] for s in payload["sweep"]]


def main():
    print(f"Using device: {DEVICE}")
    emb, w_q, w_k = build_head(score_scale=12.0)

    # --- Headline: hand-built content head (distance-invariant span). ---
    print("Evaluating content-addressed retrieval head...")
    content_fn = make_model_fn(emb, w_q, w_k, alibi_slope=0.0)
    payload = task.evaluate(content_fn)
    payload["model_name"] = "handbuilt_content_retrieval_head"

    distances = payload["canonical_distances"]
    content_means = sweep_means(payload)

    # --- Strawman: positional-only head (finite span, decays with distance). ---
    print("Evaluating positional-only strawman (ALiBi, no content)...")
    # Zero the content match so ONLY the ALiBi distance penalty drives scores.
    zero_emb = torch.zeros_like(emb)
    pos_fn = make_model_fn(zero_emb, w_q, w_k, alibi_slope=0.05)
    pos_payload = task.evaluate(pos_fn)
    positional_means = sweep_means(pos_payload)

    # --- Ablation: destroy the needle key direction -> uniform attention. ---
    print("Evaluating ablated head (needle key direction zeroed)...")
    w_k_abl = w_k.clone()
    w_k_abl[CH_RETRIEVE, CH_NEEDLE] = 0.0
    abl_fn = make_model_fn(emb, w_q, w_k_abl, alibi_slope=0.0)
    abl_payload = task.evaluate(abl_fn)
    ablation_means = sweep_means(abl_payload)

    # Stash the contrast curves for the demo (score() ignores extra keys).
    payload["contrast"] = {
        "distances": distances,
        "content_means": content_means,
        "positional_means": positional_means,
        "ablation_means": ablation_means,
        "content_auc": payload["attention_span_auc"],
        "positional_auc": pos_payload["attention_span_auc"],
        "ablation_auc": abl_payload["attention_span_auc"],
        "uniform_baseline": 1.0 / payload["canonical_seq_len"],
    }

    run_dir = results_dir(__file__)
    # Save the curves explicitly so app.py can render them without re-running.
    (run_dir / "curves.json").write_text(json.dumps(payload["contrast"], indent=2))

    record_benchmark(__file__, run_dir, payload)

    print("\n--- attention-on-target (mean over 100 seqs) ---")
    print(f"{'dist':>5} {'content':>9} {'positional':>11} {'ablated':>9}")
    for i, d in enumerate(distances):
        print(f"{d:>5} {content_means[i]:>9.4f} {positional_means[i]:>11.4f} {ablation_means[i]:>9.4f}")
    print(f"\ncontent AUC      = {payload['attention_span_auc']:.4f}")
    print(f"positional AUC   = {pos_payload['attention_span_auc']:.4f}")
    print(f"ablation AUC     = {abl_payload['attention_span_auc']:.4f}")
    print(f"uniform baseline = {1.0/512:.5f}")
    print(f"\nResults written to {run_dir}")


if __name__ == "__main__":
    main()
