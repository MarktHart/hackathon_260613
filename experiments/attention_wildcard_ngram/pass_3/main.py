import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

N_BATCH = 1024
SEQ_LEN = 16
VOCAB_SIZE = 32

# Load task metadata and the real Batch dataclass (the contract's input type).
task = load_task(__file__)
Batch = task.Batch

DEVICE = "cuda"


# -----------------------------------------------------------------
# Hand-coded wildcard-n-gram head, all compute on CUDA.
#
# Contract: model_fn(batch: Batch) -> (N, L, L) attention weights. We build a
# single head whose query for the target token matches the key for the anchor
# token, so the target's attention row peaks on the anchor position regardless
# of the (wildcard) tokens in between. Token-identity embeddings select a
# distinct feature dim per token id; a fixed Q/K interaction matrix routes
# target-token queries onto anchor-token keys. Softmax over keys keeps the rows
# normalised and finite.
# -----------------------------------------------------------------
class WildcardNgramAttn:
    def __init__(self, vocab_size: int, anchor_token: int = 1, target_token: int = 2):
        self.vocab_size = vocab_size
        self.anchor_token = anchor_token
        self.target_token = target_token
        # One-hot token embeddings (vocab, vocab).
        self.embed = torch.eye(vocab_size, dtype=torch.float32, device=DEVICE)
        # Interaction matrix W (vocab, vocab): a target-token query attending to
        # an anchor-token key scores high; everything else scores ~0.
        self.W = torch.zeros((vocab_size, vocab_size), dtype=torch.float32, device=DEVICE)
        self.W[target_token, anchor_token] = 12.0

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        # sequences: (B, L) int64 token ids in [0, vocab).
        B, L = sequences.shape
        q_tok = self.embed[sequences]              # (B, L, vocab) one-hot of query tok
        k_tok = self.embed[sequences]              # (B, L, vocab) one-hot of key tok

        # scores[b, i, j] = q_tok[b,i] @ W @ k_tok[b,j]
        # = W[tok_i, tok_j]. Compute via two matmuls.
        qW = q_tok @ self.W                        # (B, L, vocab)
        scores = qW @ k_tok.transpose(1, 2)        # (B, L, L)

        attn = torch.softmax(scores, dim=-1)       # (B, L, L), rows sum to 1
        return attn


def model_fn(batch: Batch) -> np.ndarray:
    with torch.inference_mode():
        seqs = torch.as_tensor(
            np.asarray(batch.sequences), dtype=torch.int64, device=DEVICE
        )
        seqs = seqs.clamp_(0, VOCAB_SIZE - 1)
        m = WildcardNgramAttn(
            vocab_size=VOCAB_SIZE,
            anchor_token=batch.anchor_token,
            target_token=batch.target_token,
        )
        attn = m.forward(seqs).detach().cpu().numpy().astype(np.float32)

    if attn.shape != (N_BATCH, SEQ_LEN, SEQ_LEN):
        raise ValueError(f"Expected attention (N=1024, L=16, L=16), got {attn.shape}")
    return attn


def run():
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)


if __name__ == "__main__":
    run()
