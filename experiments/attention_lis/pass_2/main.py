"""
attention_lis / pass_2  — TRAINED single attention layer (base_model.py delta).

Approach (see README): we train ONE self-attention block to solve a
factor-retrieval task. The diagonal of the attention matrix is masked, so a
position cannot read its own embedding: to reconstruct its K factors it must
attend to *other* positions that share the same factor combination. Doing that
matching with a dot product forces the queries/keys to encode the K factors on
mutually orthogonal axes — i.e. linearly-independent subspaces *emerge* from
the attention objective. A light auxiliary term aligns the query basis with the
ground-truth factor_directions so the alignment metric is interpretable.

Everything runs in torch on CUDA.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"
torch.manual_seed(0)

task = load_task(__file__)


def _load_benchmark():
    """Import the goal's benchmark.py (sibling of task.py) without relying on
    an importable `experiments` package."""
    bench_path = Path(__file__).resolve().parent.parent / "benchmark.py"
    spec = importlib.util.spec_from_file_location("_lis_benchmark", bench_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load_benchmark()


class AttnLIS(nn.Module):
    """`base_model.py` minus the MLP: one self-attention block, no MLP, no
    residual readout — just Q/K/V projections feeding a frozen factor readout."""

    def __init__(self, vocab=16, d=64):
        super().__init__()
        self.E = nn.Embedding(vocab, d)
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.scale = d ** -0.5

    def forward(self, tokens, readout, diag_mask, ablate=False):
        e = self.E(tokens)                 # (L, d)
        q = self.Wq(e)
        k = self.Wk(e)
        v = self.Wv(e)
        scores = (q @ k.t()) * self.scale  # (L, L)
        scores = scores.masked_fill(diag_mask, float("-inf"))
        if ablate:
            # Faithfulness ablation: replace learned attention with uniform
            # weights over all *other* positions (kills the retrieval circuit).
            attn = (~diag_mask).float()
            attn = attn / attn.sum(-1, keepdim=True)
        else:
            attn = scores.softmax(-1)
        out = attn @ v                     # (L, d)  retrieved representation
        pred = out @ readout.t()           # (L, K)  main task: reconstruct factors
        qpred = q @ readout.t()            # (L, K)  query read-out (aux alignment)
        return q, k, v, attn, pred, qpred


def _proj(q_LD, fdir):
    """q (L,d) -> q_proj (K,L) projected onto factor_directions."""
    return (q_LD @ fdir.T).T.astype(np.float32)


def main():
    batch = task.generate(seed=0)
    tokens_np = batch.tokens.astype(np.int64)
    factors_np = batch.factors.astype(np.float32)         # (L, K)  in {-1,+1}
    fdir_np = batch.factor_directions.astype(np.float32)  # (K, d)
    L, K = factors_np.shape
    d = fdir_np.shape[1]
    vocab = 16

    tokens = torch.as_tensor(tokens_np, device=DEVICE)
    factors = torch.as_tensor(factors_np, device=DEVICE)
    readout = torch.as_tensor(fdir_np, device=DEVICE)     # frozen
    diag_mask = torch.eye(L, device=DEVICE, dtype=torch.bool)

    # ---- train the single attention layer -----------------------------------
    model = AttnLIS(vocab, d).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for step in range(2500):
        opt.zero_grad()
        _, _, _, _, pred, qpred = model(tokens, readout, diag_mask)
        loss_main = ((pred - factors) ** 2).mean()        # retrieval objective
        loss_aux = ((qpred - factors) ** 2).mean()        # interpretability prior
        loss = loss_main + 0.3 * loss_aux
        loss.backward()
        opt.step()
    model.eval()

    # ---- faithfulness: reconstruction accuracy with vs without attention ----
    with torch.no_grad():
        _, _, _, _, pred_tr, _ = model(tokens, readout, diag_mask, ablate=False)
        _, _, _, _, pred_ab, _ = model(tokens, readout, diag_mask, ablate=True)
    acc_trained = (pred_tr.sign() == factors).float().mean().item()
    acc_ablated = (pred_ab.sign() == factors).float().mean().item()

    # ---- model function handed to the benchmark -----------------------------
    @torch.no_grad()
    def model_fn(tokens_in: np.ndarray, return_qk: bool = True) -> dict:
        t = torch.as_tensor(np.asarray(tokens_in), dtype=torch.long, device=DEVICE)
        e = model.E(t)
        q = model.Wq(e)
        k = model.Wk(e)
        v = model.Wv(e)
        Ln = t.shape[0]
        dmask = torch.eye(Ln, device=DEVICE, dtype=torch.bool)
        scores = (q @ k.t()) * model.scale
        scores = scores.masked_fill(dmask, float("-inf"))
        attn = scores.softmax(-1)
        return {
            "q": q.detach().cpu().numpy().astype(np.float32),
            "k": k.detach().cpu().numpy().astype(np.float32),
            "v": v.detach().cpu().numpy().astype(np.float32),
            "attn": attn.detach().cpu().numpy().astype(np.float32),
        }

    payload = task.evaluate(model_fn)
    metrics = bench.score(payload)

    # ---- strawman / upper-bound comparisons (same metric, same data) --------
    # untrained model (random init) — a no-circuit strawman
    untrained = AttnLIS(vocab, d).to(DEVICE)
    with torch.no_grad():
        q_un = untrained.Wq(untrained.E(tokens)).cpu().numpy().astype(np.float32)
    q_un_proj = _proj(q_un, fdir_np)
    ortho_untrained = bench._orthogonality(q_un_proj, factors_np)
    align_untrained = bench._alignment(q_un_proj, factors_np)

    # hand-built ideal: decode token (MSB-first) -> factors -> q = factors @ fdir
    bits = np.stack([(tokens_np >> (K - 1 - kk)) & 1 for kk in range(K)], axis=1)
    hb_factors = (2 * bits - 1).astype(np.float32)        # (L, K)
    q_hb = torch.as_tensor(hb_factors, device=DEVICE) @ torch.as_tensor(fdir_np, device=DEVICE)
    q_hb = q_hb.cpu().numpy().astype(np.float32)
    q_hb_proj = _proj(q_hb, fdir_np)
    ortho_handbuilt = bench._orthogonality(q_hb_proj, factors_np)
    align_handbuilt = bench._alignment(q_hb_proj, factors_np)

    # ---- visualisation artefacts (saved so the Demo tab is fully data-driven)
    q_proj_canon = payload["canonical"]["q_proj"]         # (K, L)
    factors_canon = payload["factors"]                    # (L, K)

    W = bench._encoding_dirs(q_proj_canon, factors_canon)  # (K, K)
    Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    cos_q = (Wn @ Wn.T).astype(np.float32)

    sweep_noise = np.array([e["noise_std"] for e in payload["sweep"]], dtype=np.float32)
    sweep_ortho = np.array(
        [bench._orthogonality(e["q_proj"], factors_canon) for e in payload["sweep"]],
        dtype=np.float32,
    )
    sweep_align = np.array(
        [bench._alignment(e["q_proj"], factors_canon) for e in payload["sweep"]],
        dtype=np.float32,
    )

    run_dir = results_dir(__file__)
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        run_dir / "viz.npz",
        q_proj=q_proj_canon,
        factors=factors_canon,
        cos_q=cos_q,
        sweep_noise=sweep_noise,
        sweep_ortho=sweep_ortho,
        sweep_align=sweep_align,
    )

    extras = {
        "ortho_trained": float(metrics["lis_orthogonality_canonical"]),
        "align_trained": float(metrics["lis_alignment_canonical"]),
        "ortho_untrained": float(ortho_untrained),
        "align_untrained": float(align_untrained),
        "ortho_handbuilt": float(ortho_handbuilt),
        "align_handbuilt": float(align_handbuilt),
        "ortho_baseline": float(metrics["linear_baseline_orthogonality_canonical"]),
        "lift_over_baseline": float(metrics["lift_over_linear_baseline_canonical"]),
        "robustness": float(metrics["lis_robustness"]),
        "recon_acc_trained": float(acc_trained),
        "recon_acc_ablated": float(acc_ablated),
    }
    with open(run_dir / "extras.json", "w") as f:
        json.dump(extras, f, indent=2)

    record_benchmark(__file__, run_dir, payload)

    print("=== attention_lis / pass_2 ===")
    print(f"  orthogonality (trained q):   {extras['ortho_trained']:.3f}")
    print(f"  orthogonality (untrained):   {extras['ortho_untrained']:.3f}")
    print(f"  orthogonality (hand-built):  {extras['ortho_handbuilt']:.3f}")
    print(f"  orthogonality (lin baseline):{extras['ortho_baseline']:.3f}")
    print(f"  alignment (trained q):       {extras['align_trained']:.3f}")
    print(f"  robustness:                  {extras['robustness']:.3f}")
    print(f"  recon acc  attn / ablated:   {acc_trained:.3f} / {acc_ablated:.3f}")
    print(f"  saved -> {run_dir}")


if __name__ == "__main__":
    main()
