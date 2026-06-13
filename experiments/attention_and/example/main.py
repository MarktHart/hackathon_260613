"""Example experiment. Runs from the shared workspace venv.

Invoke from the repo root:

    uv run python experiments/attention_and/example/main.py
"""

from agentic.experiments import results_dir


def main() -> None:
    out = results_dir(__file__)
    (out / "summary.txt").write_text("hello from the example experiment\n")
    print(f"wrote {out / 'summary.txt'}")


if __name__ == "__main__":
    main()
