"""
pass_3 app — visualise the hand-built QK/softmax stack-matching circuit.

Demo tab:
  (1) Attention heatmap for a chosen sequence/head, with the TRUE matching
      pairs marked, so a human can verify each ')' row's bright pixel sits on
      its matching '(' column. The attention shown is the real softmax output.
  (2) The money plot: mean attention-to-match vs nesting depth for
      Full circuit  vs  Recency-only (depth ablated)  vs  Uniform chance.
      The full circuit stays near 1.0 across depths; the ablation collapses
      with depth — causal evidence the depth feature is what does the work.

Benchmark tab: shared benchmark_panel across all attempts.
"""

import numpy as np
import torch
import gradio as gr

from agentic.experiments import benchmark_panel, load_task

DEVICE = "cuda"
N_HEADS = 4
DCAP = 18
A_DEPTH = 1.0e4
W_OPEN = 1.0e3
C_REC = 3.0
GOAL_DIR = "experiments/attention_cfg_generate"


# ---- self-contained copy of the circuit (mirrors main.build_attention) ----
def build_attention(input_ids, a_depth=A_DEPTH, w_open=W_OPEN, c_rec=C_REC):
    tokens = torch.as_tensor(input_ids, dtype=torch.long, device=DEVICE)
    B, S = tokens.shape
    is_open = (tokens == 1).float()
    is_close = (tokens == 2).float()
    signed = is_open - is_close
    c = torch.cumsum(signed, dim=1)
    m = torch.where(tokens == 1, c,
                    torch.where(tokens == 2, c + 1.0, torch.zeros_like(c)))
    m = m.clamp(0, DCAP - 1).long()
    depth_onehot = torch.zeros(B, S, DCAP, device=DEVICE)
    depth_onehot.scatter_(2, m.unsqueeze(-1), 1.0)
    pos = torch.arange(S, device=DEVICE, dtype=torch.float32)
    recency = pos.unsqueeze(0).expand(B, S)
    ones = torch.ones(B, S, 1, device=DEVICE)
    K = torch.cat([depth_onehot, is_open.unsqueeze(-1), recency.unsqueeze(-1)], dim=-1)
    Q = torch.cat([a_depth * depth_onehot, w_open * ones, c_rec * ones], dim=-1)
    scores = Q @ K.transpose(1, 2)
    causal = torch.tril(torch.ones(S, S, device=DEVICE)).bool()
    scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = attn.unsqueeze(1).expand(B, N_HEADS, S, S).contiguous()
    return attn.detach().cpu().numpy().astype(np.float32)


def _batch():
    task = load_task(__file__)
    return task.generate(seed=42)


def _sym(t):
    return {0: "·", 1: "(", 2: ")"}.get(int(t), "?")


def heatmap(seq_idx, head):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    batch = _batch()
    seq_idx = int(seq_idx) % len(batch.tokens)
    head = int(head)
    toks = batch.tokens[seq_idx]
    attn = build_attention(toks[None, :])  # [1,H,S,S]

    end = int((toks != 0).sum()) or len(toks)
    grid = attn[0, head, :end, :end]
    labels = [_sym(t) for t in toks[:end]]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(end)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(end)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Key position (attended-to)")
    ax.set_ylabel("Query position (predicting)")
    ax.set_title(f"seq {seq_idx}, head {head} — softmax attention")
    # mark true matches: at row=close, col=match-open
    for op, cp, d in batch.pairs[seq_idx]:
        if cp < end and op < end:
            ax.add_patch(plt.Rectangle((op - 0.5, cp - 0.5), 1, 1,
                                       fill=False, edgecolor="cyan", lw=1.4))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    # text summary: attention each close puts on its true match
    lines = [f"sequence: {''.join(labels)}", "",
             "close_pos -> match_open  (depth)  attn_to_match"]
    for op, cp, d in sorted(batch.pairs[seq_idx], key=lambda x: x[1]):
        if cp < end:
            val = float(np.mean(attn[0, :, cp, op]))
            lines.append(f"  {cp:>2} -> {op:>2}      (d={d})   {val:.4f}")
    return fig, "\n".join(lines)


def _sweep(a_depth):
    """Mean attention-to-match per depth for the given circuit setting."""
    batch = _batch()
    attn = build_attention(batch.tokens, a_depth=a_depth)  # [B,H,S,S]
    sums = {d: 0.0 for d in range(1, 6)}
    unif = {d: 0.0 for d in range(1, 6)}
    cnt = {d: 0 for d in range(1, 6)}
    for b in range(len(batch.tokens)):
        for op, cp, d in batch.pairs[b]:
            if d in sums:
                sums[d] += float(np.mean(attn[b, :, cp, op]))
                unif[d] += 1.0 / (cp + 1)  # no PAD: prefix length = cp+1
                cnt[d] += 1
    mean = {d: (sums[d] / cnt[d] if cnt[d] else 0.0) for d in sums}
    unif_mean = {d: (unif[d] / cnt[d] if cnt[d] else 0.0) for d in sums}
    return mean, unif_mean


def comparison():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full, unif = _sweep(A_DEPTH)
    nodepth, _ = _sweep(0.0)
    depths = list(range(1, 6))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(depths, [full[d] for d in depths], "o-", color="#2e7d32",
            lw=2.5, ms=8, label="Full QK circuit (stack)")
    ax.plot(depths, [nodepth[d] for d in depths], "s--", color="#c62828",
            lw=2, ms=7, label="Recency-only (depth ablated)")
    ax.plot(depths, [unif[d] for d in depths], "^:", color="#616161",
            lw=2, ms=7, label="Uniform chance baseline")
    ax.set_xlabel("Nesting depth of the closing ')'")
    ax.set_ylabel("Mean attention on matching '('")
    ax.set_xticks(depths)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Stack attention vs depth — circuit, ablation, baseline")
    ax.grid(alpha=0.3)
    ax.legend(loc="center right")
    plt.tight_layout()

    txt = ["depth |  full  | recency-only | uniform"]
    for d in depths:
        txt.append(f"  {d}   | {full[d]:.3f} |    {nodepth[d]:.3f}    | {unif[d]:.3f}")
    return fig, "\n".join(txt)


with gr.Blocks() as demo:
    gr.Markdown(
        "## Stack attention in Dyck-1 generation (pass_3)\n"
        "A **hand-set QK circuit**: attention to the matching `(` *emerges* from "
        "`softmax(Q·Kᵀ)` over depth-match + is-open + recency features — not written in."
    )

    with gr.Tabs():
        with gr.TabItem("Demo"):
            gr.Markdown(
                "**Top:** per-sequence softmax heatmap — cyan boxes mark the true "
                "matching pairs; a bright pixel inside each box = the `)` attends to "
                "its `(`.  **Bottom:** the claim — full circuit holds near 1.0 across "
                "depth, while removing the depth feature (recency-only) collapses."
            )
            with gr.Row():
                seq = gr.Slider(0, 255, value=0, step=1, label="Sequence index")
                head = gr.Slider(0, N_HEADS - 1, value=0, step=1, label="Head")
                btn = gr.Button("Show", variant="primary")
            with gr.Row():
                hm = gr.Plot(label="Attention heatmap")
                hm_txt = gr.Textbox(label="Attention on true match", lines=18)
            gr.Markdown("### Depth sweep — circuit vs ablation vs baseline")
            with gr.Row():
                cmp_plot = gr.Plot(label="Mean attn-to-match vs depth")
                cmp_txt = gr.Textbox(label="Per-depth values", lines=8)

            btn.click(heatmap, inputs=[seq, head], outputs=[hm, hm_txt])
            demo.load(heatmap, inputs=[seq, head], outputs=[hm, hm_txt])
            demo.load(comparison, inputs=None, outputs=[cmp_plot, cmp_txt])

        with gr.TabItem("Benchmark"):
            gr.Markdown("### Leaderboard & metric history across attempts")
            benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
