"""`agentic <task>` — run a task and print the streamed messages."""

import asyncio
import json

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


@app.command()
def pipeline(
    slug: str | None = typer.Option(
        None, "--slug", "-s", help="Goal slug to run. If omitted, picks the next pending block."
    ),
    skip_solver: bool = typer.Option(
        False, "--skip-solver", help="Stop after the benchmark is written and reviewed."
    ),
    skip_jury: bool = typer.Option(
        False, "--skip-jury", help="Stop after the solver produces an attempt."
    ),
    force: bool = typer.Option(False, "--force", help="Re-run even if the slug is already graded."),
) -> None:
    """Run the picker → reviewer → solver → jury pipeline."""
    from agentic.pipeline import run_pipeline

    result = asyncio.run(
        run_pipeline(slug=slug, skip_solver=skip_solver, skip_jury=skip_jury, force=force)
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("pipeline-multi")
def pipeline_multi(
    count: int | None = typer.Option(
        None, "-c", "--count", help="Number of pending slugs to process (default: all pending)."
    ),
    n_concurrent: int | None = typer.Option(
        None,
        "-n",
        "--concurrent",
        help="Cap concurrent LLM stages. Leave unset to rely on the GPU pool.",
    ),
    skip_solver: bool = typer.Option(False, "--skip-solver"),
    skip_jury: bool = typer.Option(False, "--skip-jury"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Run pipelines for multiple pending slugs concurrently."""
    from agentic.pipeline import run_pipeline_multi

    result = asyncio.run(
        run_pipeline_multi(
            count=count,
            n_concurrent=n_concurrent,
            skip_solver=skip_solver,
            skip_jury=skip_jury,
            force=force,
        )
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("events")
def show_events(
    last: int = typer.Option(20, "--last", "-n", help="How many recent events to print."),
) -> None:
    """Tail the pipeline event log."""
    from agentic.events import read_events

    events = read_events()
    for record in events[-last:]:
        typer.echo(json.dumps(record))


if __name__ == "__main__":
    app()
