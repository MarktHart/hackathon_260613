"""Hand-built 2-layer induction circuit (pass_4) — the REAL mechanism.

Unlike pass_3 (which ran generic blocks and then *bypassed* them with a
separate gather), here the next-token logits genuinely flow through the
attention + unembedding of a tiny 2-block attention-only transformer. Every
weight is hand-set; nothing is trained. The circuit is the textbook two-head
induction motif (Elhage et al.):

  Layer 0 — PREVIOUS-TOKEN HEAD (positional).
      Each position r attends to position r-1 and copies that token's identity
      into a dedicated "prev" subspace of the residual stream.
      After layer 0:  prev[r] = onehot(token[r-1]).

  Layer 1 — INDUCTION HEAD (content match + copy).
      Query at p reads the current token:        Q[p] = onehot(token[p]).
      Key   at r reads the prev subspace:         K[r] = onehot(token[r-1]).
      So p attends to the position r where token[r-1] == token[p]  ->  r = q+1,
      i.e. one past the *earlier* occurrence q of the current token.
      Value copies that position's token:         V[r] = onehot(token[r]).
      Output written to an "out" subspace; the unembedding reads it.
      Result: predict token[q+1] = A_{j+1}.  Pure induction.

This is `base_model.py` reduced to its skeleton: token embedding + two
attention heads + residual stream + an unembedding, NO MLP (the MLP is dead
weight for this task, so it is stripped — see README). The residual stream is
partitioned into three 128-wide one-hot slots: content / prev / out.

FAITHFULNESS CHECK (recorded for the demo): re-running with the layer-0 output
projection zeroed (`ablate_prev=True`) removes the previous-token head, so the
induction key collapses to zero, attention goes uniform, and accuracy falls to
the uniform baseline. The behaviour is *caused* by the circuit, not bolted on.
"""

import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # guaranteed by the pipeline; no CPU fallback

task = load_task(__file__)
VOCAB = task.VOCAB_SIZE      # 128
SEQ_LEN = task.SEQ_LEN       # 192

# Residual-stream layout: [ content (128) | prev (128) | out (128) ]
CONTENT = (0, VOCAB)
PREV = (VOCAB, 2 * VOCAB)
OUT = (2 * VOCAB, 3 * VOCAB)
D_MODEL = 3 * VOCAB

TEMP = 30.0    # induction-head inverse-temperature (sharpens one-hot matches)
SCALE = 20.0   # unembedding gain (confident logits on the copied token)


def build_weights():
    """Hand-set every weight as a torch tensor on CUDA. No training."""
    z = lambda *s: torch.zeros(*s, device=DEVICE)
    eye = torch.eye(VOCAB, device=DEVICE)

    # Token embedding: token t -> one-hot in the content slot.
    W_E = z(VOCAB, D_MODEL)
    W_E[:, CONTENT[0]:CONTENT[1]] = eye

    # --- Layer 0: previous-token head ---
    # Value reads the content slot; output writes into the prev slot.
    Wv0 = z(D_MODEL, VOCAB)
    Wv0[CONTENT[0]:CONTENT[1], :] = eye
    Wo0 = z(VOCAB, D_MODEL)
    Wo0[:, PREV[0]:PREV[1]] = eye

    # --- Layer 1: induction head ---
    Wq1 = z(D_MODEL, VOCAB)           # query reads content slot
    Wq1[CONTENT[0]:CONTENT[1], :] = eye
    Wk1 = z(D_MODEL, VOCAB)           # key reads prev slot
    Wk1[PREV[0]:PREV[1], :] = eye
    Wv1 = z(D_MODEL, VOCAB)           # value reads content slot
    Wv1[CONTENT[0]:CONTENT[1], :] = eye
    Wo1 = z(VOCAB, D_MODEL)           # output writes out slot
    Wo1[:, OUT[0]:OUT[1]] = eye

    # Unembedding: read out slot, gain SCALE.
    W_U = z(D_MODEL, VOCAB)
    W_U[OUT[0]:OUT[1], :] = eye * SCALE

    # Layer-0 positional pattern: position r attends to r-1 (r=0 -> itself).
    neg = torch.full((SEQ_LEN, SEQ_LEN), -1e9, device=DEVICE)
    for r in range(SEQ_LEN):
        neg[r, r - 1 if r >= 1 else 0] = 0.0
    pos_bias = neg

    # Layer-1 causal mask: position p may attend to r <= p.
    idx = torch.arange(SEQ_LEN, device=DEVICE)
    causal = (idx.unsqueeze(1) >= idx.unsqueeze(0))  # (p, r) True if r<=p

    return dict(W_E=W_E, Wv0=Wv0, Wo0=Wo0, Wq1=Wq1, Wk1=Wk1, Wv1=Wv1,
                Wo1=Wo1, W_U=W_U, pos_bias=pos_bias, causal=causal)


def forward_logits(ids: torch.Tensor, W: dict, ablate_prev: bool = False):
    """Genuine forward pass: embedding -> 2 attention layers -> unembedding."""
    B, S = ids.shape
    x = W["W_E"][ids]  # (B, S, D)

    # Layer 0 — previous-token head (fixed positional attention).
    A0 = torch.softmax(W["pos_bias"], dim=-1)       # (S, S)
    V0 = x @ W["Wv0"]                                # (B, S, 128)
    head0 = torch.matmul(A0, V0)                     # (B, S, 128)
    if not ablate_prev:
        x = x + head0 @ W["Wo0"]                     # write prev slot

    # Layer 1 — induction head (content/prev match, copy content).
    Q = x @ W["Wq1"]                                 # (B, S, 128)
    K = x @ W["Wk1"]                                 # (B, S, 128)
    scores = torch.matmul(Q, K.transpose(1, 2)) * TEMP   # (B, S, S)
    scores = scores.masked_fill(~W["causal"], -1e9)
    A1 = torch.softmax(scores, dim=-1)
    V1 = x @ W["Wv1"]
    head1 = torch.matmul(A1, V1)                     # (B, S, 128)
    x = x + head1 @ W["Wo1"]                         # write out slot

    logits = x @ W["W_U"]                            # (B, S, vocab)
    return logits


def make_model_fn(W: dict, ablate_prev: bool = False):
    def model_fn(input_ids: np.ndarray) -> np.ndarray:
        ids = torch.as_tensor(input_ids, dtype=torch.int64, device=DEVICE)
        with torch.no_grad():
            logits = forward_logits(ids, W, ablate_prev=ablate_prev)
        return logits.detach().cpu().numpy().astype(np.float32)
    return model_fn


def _sweep_accs(payload):
    return {int(r["distance"]): float(r["accuracy"]) for r in payload["sweep"]}


if __name__ == "__main__":
    W = build_weights()

    # Headline: the full hand-built induction circuit.
    full_fn = make_model_fn(W, ablate_prev=False)
    payload = task.evaluate(full_fn)

    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    # Faithfulness: ablate the previous-token head -> induction must collapse.
    abl_fn = make_model_fn(W, ablate_prev=True)
    abl_payload = task.evaluate(abl_fn)

    uniform_acc = 1.0 / VOCAB
    demo = {
        "vocab_size": VOCAB,
        "seq_len": SEQ_LEN,
        "distances": list(_sweep_accs(payload).keys()),
        "full": {
            "aggregate_accuracy": payload["aggregate"]["accuracy"],
            "aggregate_ce_loss": payload["aggregate"]["ce_loss"],
            "by_distance": _sweep_accs(payload),
        },
        "ablated_prev_head": {
            "aggregate_accuracy": abl_payload["aggregate"]["accuracy"],
            "aggregate_ce_loss": abl_payload["aggregate"]["ce_loss"],
            "by_distance": _sweep_accs(abl_payload),
        },
        "uniform_baseline_accuracy": uniform_acc,
    }
    with open(run_dir / "demo.json", "w") as f:
        json.dump(demo, f, indent=2)

    print(f"Full induction accuracy:    {payload['aggregate']['accuracy']:.4f}")
    print(f"Ablated (no prev-head) acc: {abl_payload['aggregate']['accuracy']:.4f}")
    print(f"Uniform baseline:           {uniform_acc:.4f}")
    print(f"Benchmark recorded to {run_dir / 'benchmark.json'}")
