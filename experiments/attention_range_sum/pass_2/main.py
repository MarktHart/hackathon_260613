import torch
import torch.nn.functional as F
from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)
DEVICE = "cuda"
device = torch.device(DEVICE)

# Hand-built range-sum head, constructed deterministically (no checkpoint).
#
# Contract: model_fn(input_ids: (L,) int, start, end) -> float, returning the
# sum of input_ids[start:end]. We implement this as an attention head: a window
# mask gives uniform attention weight 1/k over the k tokens in [start, end);
# the attention-weighted mean of the token *values* is therefore mean(window),
# and multiplying by the window length k recovers the sum. All compute on CUDA.


def model_fn(input_ids, start: int, end: int) -> float:
    seq = torch.as_tensor(input_ids, dtype=torch.float32, device=device)  # (L,)
    L = seq.shape[0]
    start = int(start)
    end = int(end)
    k = max(end - start, 1)

    pos = torch.arange(L, device=device)
    in_window = (pos >= start) & (pos < end)  # (L,) bool

    # Attention logits: 0 inside the window, -inf outside -> uniform over window.
    logits = torch.where(
        in_window,
        torch.zeros(L, device=device),
        torch.full((L,), float("-inf"), device=device),
    )
    attn = F.softmax(logits, dim=0)  # (L,), uniform 1/k on window, 0 elsewhere

    weighted_mean = torch.dot(attn, seq)        # mean of window token values
    total = weighted_mean * float(k)            # mean * count = sum
    return float(total.detach().cpu().item())


payload = task.evaluate(model_fn)
record_benchmark(__file__, results_dir(__file__), payload)
