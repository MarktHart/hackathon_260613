"""
attention_sign_threshold / pass_4  --  TRAINED attention QK circuit + causal ablation.

Delta from base_model.py: keep ONLY one attention head's QK pathway,
    logit(q, k) = q^T M k   with a single learned bilinear form M (d x d).
No value/output projection, no MLP, no softmax-over-sequence (each pair is
scored independently). M is NOT hand-set; it is discovered by gradient
descent on a sign-classification objective (attend iff cos(q,k) > 0).

We then (a) check faithfulness by measuring how close learned M is to a
scalar*Identity (i.e. it rediscovered the dot product), and (b) run a causal
ablation that zeroes the learned circuit and shows the sweep goes flat.
All real compute runs in torch on cuda.
"""
import json
import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)
DEVICE = "cuda"
D = 64


def _unit(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


def train_qk_matrix(steps=400, batch=2048, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    M = torch.zeros(D, D, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([M], lr=0.05)
    bce = torch.nn.BCEWithLogitsLoss()
    curve = []
    for t in range(steps):
        q = _unit(torch.randn(batch, D, device=DEVICE, generator=g))
        k = _unit(torch.randn(batch, D, device=DEVICE, generator=g))
        label = ((q * k).sum(-1) > 0).float()
        logit = torch.einsum("bd,de,be->b", q, M, k)
        loss = bce(logit, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if t % 20 == 0 or t == steps - 1:
            curve.append([t, float(loss)])
    return M.detach(), curve


def make_model_fn(M):
    Mt = M.to(DEVICE)

    def model_fn(queries_np, keys_np):
        q = torch.as_tensor(queries_np, dtype=torch.float32, device=DEVICE)
        k = torch.as_tensor(keys_np, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            logit = torch.einsum("bd,de,be->b", q, Mt, k)
        return logit.detach().cpu().numpy().astype(np.float32)

    return model_fn


def sweep_means(model_fn):
    payload = task.evaluate(model_fn)
    cos = [r["cosine"] for r in payload["sweep"]]
    mean = [r["mean_attention"] for r in payload["sweep"]]
    return payload, cos, mean


def main():
    print("Training bilinear QK sign-detector on", DEVICE, "...")
    M, curve = train_qk_matrix()

    payload, cos, mean = sweep_means(make_model_fn(M))

    # faithfulness: did training rediscover the dot product?  M ~ s*I ?
    I = torch.eye(D, device=DEVICE)
    align = float(torch.nn.functional.cosine_similarity(M.flatten(), I.flatten(), dim=0))
    diag = float(M.diagonal().mean()) if hasattr(M, "diagonal") else float(torch.diag(M).mean())

    # causal ablations through the SAME canonical evaluator
    _, _, mean_zero = sweep_means(make_model_fn(torch.zeros_like(M)))
    M_off = M - torch.diag(torch.diag(M))
    _, _, mean_off = sweep_means(make_model_fn(M_off))

    payload["model_info"] = {
        "name": "trained_bilinear_qk_head",
        "type": "head_hook",
        "notes": (f"Single learned bilinear attention score q^T M k, no MLP. "
                  f"M discovered by BCE on sign(q.k). cos(vec(M),vec(I))={align:.3f}."),
    }

    run_dir = results_dir(__file__)
    np.save(run_dir / "M.npy", M.cpu().numpy())
    with open(run_dir / "artifacts.json", "w") as f:
        json.dump({
            "cos": cos,
            "mean_trained": mean,
            "mean_ablate_zero": mean_zero,
            "mean_ablate_offdiag": mean_off,
            "linear_baseline": [max(0.0, c) for c in cos],
            "train_curve": curve,
            "M_identity_alignment": align,
            "M_mean_diag": diag,
        }, f)
    record_benchmark(__file__, run_dir, payload)
    print(f"Done. cos(M,I)={align:.3f} mean_diag={diag:.3f}. Written to {run_dir}")


if __name__ == "__main__":
    main()
