"""Run an attempt's `main.py` and assert it actually used the GPU.

The pipeline launches attempts through this guard instead of invoking
`python main.py` directly. A reserved GPU slot that an attempt never touches
is pure waste — and silently passing a pure-NumPy/CPU attempt defeats the
whole point of a GPU benchmark. So we execute `main.py` in *this* process via
`runpy` and then read torch's peak CUDA allocation: any real CUDA tensor bumps
`torch.cuda.max_memory_allocated`, so a zero reading after `main.py` finishes
means the model never ran on the GPU, and we exit non-zero.

Usage (matches a plain `python main.py ...` invocation):

    python -m agentic.gpu_guard <path/to/main.py> [args...]

The check only fires when a CUDA device is actually visible (i.e. the pipeline
allocated a slot and set `CUDA_VISIBLE_DEVICES`). With no GPU present we run
`main.py` unchanged and skip the assertion, so the guard is a no-op on
CPU-only machines rather than a hard failure.
"""

from __future__ import annotations

import runpy
import sys

EXIT_USAGE = 2
EXIT_NO_GPU_USED = 3


def main(argv: list[str]) -> int:
    if not argv:
        print("gpu_guard: missing path to main.py", file=sys.stderr)
        return EXIT_USAGE

    main_path = argv[0]

    import torch

    # Peak allocation starts at 0 for a fresh process and importing torch here
    # allocates nothing, so we don't need to reset stats — any CUDA tensor the
    # attempt creates will be reflected in `max_memory_allocated`.
    require_gpu = torch.cuda.is_available()

    # Present the attempt with the argv it would see under `python main.py ...`
    # so its own CLI parsing is unaffected by the guard wrapper.
    sys.argv = [main_path, *argv[1:]]
    runpy.run_path(main_path, run_name="__main__")

    if not require_gpu:
        print("gpu_guard: no CUDA device visible; skipping GPU-usage check")
        return 0

    peak = sum(
        torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
    )
    if peak == 0:
        print(
            "GPU GUARD FAILED: main.py finished without allocating any CUDA memory. "
            "This attempt ran entirely on the CPU. The model_fn must execute on the "
            "GPU via torch (build/move tensors with device='cuda'). Pure-NumPy/CPU "
            "attempts are rejected — see README_EXPERIMENT.md.",
            file=sys.stderr,
        )
        return EXIT_NO_GPU_USED

    print(f"gpu_guard: ok, peak CUDA allocation {peak / (1024 * 1024):.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
