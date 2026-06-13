"""`agentic <task>` — run a task and print the streamed messages."""

import asyncio

import typer

from agentic.runner import run_task

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    task: str = typer.Argument(..., help="The task description for the orchestrator."),
) -> None:
    """Run a task through the agentic orchestrator."""

    async def _go() -> None:
        async for message in run_task(task):
            typer.echo(message)

    asyncio.run(_go())


@app.command("run-goal")
def run_goal(
    goal_path: str = typer.Argument(
        ..., help="Path to the goal dir, e.g. experiments/attention_and"
    ),
    attempt_name: str | None = typer.Option(
        None,
        "--attempt",
        "-a",
        help="Suggested attempt name. The experimenter may override.",
    ),
) -> None:
    """Dispatch the `experimenter` sub-agent at a mech-interp goal directory."""
    prompt = (
        f"Use the `experimenter` sub-agent to implement an attempt at the "
        f"mech-interp goal located at `{goal_path}`. It should follow "
        f"`README_EXPERIMENT.md` and produce `main.py`, `app.py`, and "
        f"`README.md` inside `{goal_path}/<attempt_name>/`."
    )
    if attempt_name:
        prompt += f" Suggested attempt name: `{attempt_name}`."

    async def _go() -> None:
        async for message in run_task(prompt):
            typer.echo(message)

    asyncio.run(_go())


if __name__ == "__main__":
    app()
