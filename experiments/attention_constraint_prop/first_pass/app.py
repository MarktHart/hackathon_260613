import gradio as gr
from agentic.experiments import benchmark_panel, results_dir

import experiments.attention_constraint_prop.task as task
import experiments.attention_constraint_prop.benchmark as bench

# ---- Demo ---------------------------------------------------------------
def demo_run(*_) -> gr.DataFrame:
    """Compute demo results for a single batch and return a Pandas DataFrame for the leaderboard."""
    # The model_fn used in main.py is a thin wrapper around torch.  Here we recompute attention
    # weights for the demo batch, which is identical to the one used in the benchmark.
    # We use the same tiny 1-layer self-attention block on GPU.

    device = torch.device("cuda")
    B, S = task.SEQ_LEN, task.NUM_SEQUENCES

    # build toy model
    C, H = 256, 8
    net = torch.nn.ModuleDict({
        "emb": torch.nn.Embedding(task.VOCAB_SIZE, C).to(device),
        "qkv": torch.nn.Linear(C, 3 * C // H, bias=False).to(device),
        "atten": torch.nn.MultiheadAttention(C, H, batch_first=True, dropout=0.0).to(device),
        "out": torch.nn.Linear(C, C, bias=False).to(device),
    })

    # token embeddings on GPU
    rng = task.np.random.default_rng(seed=0)    # reproducible
    input_ids = np.stack(
        [rng.integers(0, task.VOCAB_SIZE, size=S).astype(np.int32) for _ in range(B)]
    )
    ids = torch.as_tensor(input_ids, dtype=torch.int64, device=device)    # [B, S]

    # compute attention weights
    emb = net["emb"](ids)                                           # [B, S, C]
    qkv = net["qkv"](emb)                                           # [B, S, 3 * H, C//H]
    qkv = qkv.view(B, S, 3, H, C // H).permute(2, 0, 1, 3, 4)       # [3, B, S, H, C//H]
    q, k, v = qkv.unbind(0)                                     # each [B, S, H, C//H]
    attn_weights = torch.einsum('bshd,bseh->bhsd', q, k)              # [B, H, S, S]
    attn_weights = attn_weights / torch.sqrt(torch.tensor(C // H, device=device))
    attn_weights = torch.softmax(attn_weights, dim=-1)                # [B, H, S, S]

    # reshape for model_fn shape [B, L, H, S, S] (L=1 for a single layer)
    attn_weights = attn_weights.unsqueeze(1).cpu().numpy().astype(np.float32)    # [B, 1, H, S, S]

    # generate constrained entries (directed pairs) for this batch
    batch = task.Batch(input_ids=input_ids, constraints=task._make_constraints(input_ids.shape, seed=0))
    flat = task._flatten_constraints(batch)

    payload = {
        "version": 1,
        " config": {
            "seq_len": batch.seq_len,
            "num_sequences": batch.num_sequences,
            "constraint_types": batch.constraint_types,
            "canonical_distance": batch.canonical_distance,
            "seed": batch.seed,
        },
        "model_info": {"n_layers": 1, "n_heads": H},
        "sweep": [],
    }

    # compute alignment per distance slice
    for d in sorted(np.unique(flat["d"])):
        mask = flat["d"] == d
        if not mask.any():
            continue
        bb, ii, jj = flat["b"][mask], flat["i"][mask], flat["j"][mask]
        # get mean attention weight for each head at this distance
        heads = []
        head_aligns = []
        for l in range(1):        # only one layer in our demo net
            for h in range(H):
                vals = attn_weights[bb, l, h, ii, jj]
                a = float(np.mean(vals))
                heads.append({"layer": int(l), "head": int(h), "alignment": a})
                head_aligns.append(a)
        head_aligns = np.array(head_aligns, dtype=np.float64)
        best_l, best_h = divmod(np.argmax(head_aligns), H)
        payload["sweep"].append({
            "distance": int(d),
            "n_entries": int(bb.size),
            "heads": heads,
            "mean_alignment": float(np.mean(head_aligns)),
            "max_alignment": float(np.max(head_aligns)),
            "best_head": {
                "layer": int(best_l),
                "head": int(best_h),
                "alignment": float(head_aligns[best_l]),
            },
        })

    # benchmark metrics (just a convenience for our own visual checks)
    metrics = bench.score(payload)

    def _row(d: int) -> dict:
        return {
            "distance": d,
            "n_entries": len([r for r in payload['sweep'] if r['distance'] == d]) if d in [r['distance'] for r in payload['sweep']] else 0,
            "mean_alignment": payload['sweep'][next(i for i, r in enumerate(payload['sweep']) if r['distance'] == d)]['mean_alignment'] if d == payload['sweep'][0]['distance'] else 0.0,
            "max_alignment": payload['sweep'][next(i for i, r in enumerate(payload['sweep']) if r['distance'] == d)]['max_alignment'] if d == payload['sweep'][0]['distance'] else 0.0,
            "best_head_layer": payload['sweep'][next(i for i, r in enumerate(payload['sweep']) if r['distance'] == d)]['best_head']['layer'] if d == payload['sweep'][0]['distance'] else -1,
            "best_head_head": payload['sweep'][next(i for i, r in enumerate(payload['sweep']) if r['distance'] == d)]['best_head']['head'] if d == payload['sweep'][0]['distance'] else -1,
        }
    dists = sorted(int(r['distance']) for r in payload['sweep'])
    df = gr.DataFrame([_row(d) for d in dists], columns=[
        "distance", "n_entries", "mean_alignment", "max_alignment", "best_head_layer", "best_head_head",
    ].sort())
    return df

# ---- UI ---------------------------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown("""
    # Attention Constraint Propagation (first_pass)

    Do attention heads propagate bracket-style constraints with fidelity that falls off with
    positional distance? This demo shows a tiny 1-layer self-attention model on a synthetic
    bracket batch.
    """)
    with gr.Tab("Demo"):
        with gr.Blocks():
            demo_btn = gr.Button("Run demo (uniform attention baseline)")
            demo_out = gr.DataFrame(visible=False)
            demo_btn.click(
                demo_run,
                outputs=demo_out
            )
            gr.Label("The computed DataFrame shows per-distance alignment statistics for each head in the demo model.")
    with gr.Tab("Benchmark"):
        # Load from the most recent run in results/ under this experiment
        recent_run = results_dir(__file__)
        recent_path = f"{recent_run}/benchmark.json"
        benchmark_panel("experiments/attention_constraint_prop", recent_path)

if __name__ == "__main__":
    demo.launch()