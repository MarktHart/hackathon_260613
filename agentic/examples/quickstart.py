"""Minimal example: ask the orchestrator to consult an external LLM."""

import asyncio

from agentic.runner import run_task


async def main() -> None:
    prompt = (
        "Use the external-dispatcher sub-agent to ask an external LLM: "
        "'In one sentence, what is the bias-variance tradeoff?' "
        "Return the answer verbatim."
    )
    async for message in run_task(prompt):
        print(message)


if __name__ == "__main__":
    asyncio.run(main())
