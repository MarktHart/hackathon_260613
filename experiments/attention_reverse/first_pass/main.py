"""attention_reverse / first_pass — hand-built reversal attention head.

A single attention head with NO learned weights. We replace base_model.py's
RoPE positional scheme with an *exact discrete-Fourier mirror encoding* so that
the query at position i attends exactly to key position L-1-i, for ANY length L.

Mechanism (all on the GPU, expressed as torch tensors):

  For a length-L slice, build position features over m = 0..L-1 frequencies:

      phi(p) = [cos(2*pi*m*p/L), sin(2*pi*m*p/L)]   (m = 0..L-1)   in R^{2L}

  Keys use phi(j); queries use phi(L-1-i). Then

      q_i . k_j = sum_m cos(2*pi*m*((L-1-i) - j)/L)
                = L   if  j == L-1-i  (mod L),  else 0,

  the exact discrete delta kernel. A temperature * softmax turns this into a
  near one-hot attention pattern on the mirror position. The value vectors are
  one-hot token embeddings, so the attended output at position i is (almost
  exactly) the one-hot of the token at L-1-i — i.e. the reversed token. Those
  are returned directly as logits.

Because the construction is parametric in L (frequencies scale with the actual
sequence length read off the input), the head extrapolates perfectly to lengths
it never "saw": there is no length-16 lookup to memorise.

This is the smallest delta from base_model.py: one Attention block, no MLP, no
causal mask, learned QKV/positional parts replaced by hand-set tensors.
"""
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU


def _pos_features(seq_len: int) -> torch.Tensor:
    """Exact-delta Fourier position features phi(p), shape (seq_len, 2*seq_len)."""
    p = torch.arange(seq_len, device=DEVICE, dtype=torch.float32)          # (L,)
    m = torch.arange(seq_len, device=DEVICE, dtype=torch.float32)          # (L,)
    ang = 2.0 * torch.pi * torch.outer(p, m) / seq_len                     # (L, L)
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)            # (L, 2L)


def make_model_fn(vocab_size: int, beta: float = 1.0):
    """Return a hand-built reversal model_fn.

    beta sharpens the (already peaked) softmax; beta=1 already gives ~0.997
    mirror mass at L=8 and ->1 for larger L, but we keep a knob for the demo.
    """

    def model_fn(tokens: np.ndarray):
        toks = torch.as_tensor(np.asarray(tokens), dtype=torch.long, device=DEVICE)
        n, seq_len = toks.shape

        phi = _pos_features(seq_len)                       # (L, 2L)
        mirror = torch.arange(seq_len - 1, -1, -1, device=DEVICE)  # L-1-i
        q = phi[mirror]                                    # query at i wants pos L-1-i
        k = phi                                            # key at pos j

        scores = (q @ k.t()) * beta                        # (L, L), peak L at mirror
        attn = torch.softmax(scores, dim=-1)               # (L, L) near one-hot
        attn_b = attn.unsqueeze(0).expand(n, seq_len, seq_len)  # (n, L, L)

        # Value = one-hot token embedding; attended output ~ one-hot of mirror token.
        values = torch.nn.functional.one_hot(toks, vocab_size).float()  # (n, L, V)
        logits = torch.bmm(attn_b, values)                              # (n, L, V)

        return logits.detach().cpu().numpy(), attn_b.detach().cpu().numpy()

    return model_fn


def main():
    task = load_task(__file__)
    run_dir = results_dir(__file__)

    vocab_size = task.VOCAB_SIZE
    model_fn = make_model_fn(vocab_size, beta=1.0)

    payload = task.evaluate(model_fn)
    record_benchmark(__file__, run_dir, payload)

    # Save a small artefact for the demo: the attention pattern at canonical len.
    L = task.CANONICAL_SEQ_LEN
    demo_tokens = np.arange(L, dtype=np.int64)[None, :] % vocab_size
    _, attn = model_fn(demo_tokens)
    np.save(run_dir / "canonical_attn.npy", attn[0])
    np.save(run_dir / "seq_len_sweep.npy", np.array(task.SEQ_LEN_SWEEP))

    print("Saved benchmark + artefacts to", run_dir)
    for rec in payload["sweep"]:
        print(
            f"  L={rec['seq_len']:3d}  acc={rec['accuracy']:.3f} "
            f"mirror_mass={rec['mirror_attn_mass']:.3f}"
        )


if __name__ == "__main__":
    main()
