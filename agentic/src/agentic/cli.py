"""`agentic <command>` — drive the picker → reviewer → solver → jury pipeline."""

import asyncio
import json

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


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
    """Run the picker → reviewer → solver → jury pipeline for one slug."""
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


@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8080, "--port", "-p", help="Port to serve on."),
) -> None:
    """Serve the live web dashboard (tails the event log; read-only)."""
    from agentic.dashboard.server import serve

    typer.echo(f"dashboard → http://{host}:{port}")
    serve(host=host, port=port)


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
