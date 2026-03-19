"""tractable register <config_path> — onboard a repository into the system.

Loads a Python registration config file, validates it as a
:class:`RepositoryRegistration`, runs :class:`GraphConstructionPipeline`
to ingest the repo into FalkorDB, and reports the result.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from typing import Any

import structlog
import typer
from rich.table import Table

from tractable.cli.output import console, print_error, print_success
from tractable.types.config import RepositoryRegistration

log = structlog.get_logger(__name__)

register_app = typer.Typer(help="Register a repository with tractable.")


def _load_registration(config_path: Path) -> RepositoryRegistration:
    """Load a Python config file and return the first RepositoryRegistration found."""
    spec = importlib.util.spec_from_file_location("_tractable_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from {config_path}")

    module = types.ModuleType("_tractable_config")
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    for _name, obj in inspect.getmembers(module):
        if isinstance(obj, RepositoryRegistration):
            return obj

    raise ValueError(
        f"No RepositoryRegistration instance found in {config_path}. "
        "Define one at module level: registration = RepositoryRegistration(...)"
    )


async def _run_ingest(registration: RepositoryRegistration) -> dict[str, Any]:
    """Run the ingestion pipeline and return IngestResult fields."""
    from tractable.graph.client import FalkorDBClient
    from tractable.graph.temporal_graph import FalkorDBTemporalGraph
    from tractable.parsing.parsers.typescript_parser import TypeScriptParser
    from tractable.parsing.pipeline import GraphConstructionPipeline

    graph_client = FalkorDBClient()
    graph = FalkorDBTemporalGraph(client=graph_client)

    pipeline = GraphConstructionPipeline(extra_parsers=[TypeScriptParser()])
    result = await pipeline.ingest_repository(registration=registration, graph=graph)
    return {
        "files_parsed": result.files_parsed,
        "entities_created": result.entities_created,
        "relationships_created": result.relationships_created,
        "duration_seconds": result.duration_seconds,
        "errors": result.errors,
    }


@register_app.callback(invoke_without_command=True)
def register(
    ctx: typer.Context,
    config_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to a Python file containing a RepositoryRegistration instance.",
    ),
) -> None:
    """Register a repository: load config, validate, and ingest into the graph."""
    if ctx.invoked_subcommand is not None:
        return

    # Validate file exists
    if not config_path.exists():
        print_error(f"Config file not found: {config_path}")
        raise typer.Exit(code=1)

    if not config_path.is_file():
        print_error(f"Path is not a file: {config_path}")
        raise typer.Exit(code=1)

    # Load and validate RepositoryRegistration
    try:
        registration = _load_registration(config_path)
    except (ValueError, Exception) as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc

    # Print registration summary
    summary_table = Table(show_header=False, box=None, padding=(0, 1))
    summary_table.add_row("[bold]Repo[/bold]", registration.name)
    summary_table.add_row("[bold]Git URL[/bold]", registration.git_url)
    summary_table.add_row("[bold]Template[/bold]", registration.agent_template)
    summary_table.add_row("[bold]Autonomy[/bold]", registration.autonomy_level.value)
    console.print("\n[bold cyan]Registration Summary[/bold cyan]")
    console.print(summary_table)
    console.print()

    # Run ingestion
    console.print("[bold]Running ingestion pipeline...[/bold]")
    try:
        result = asyncio.run(_run_ingest(registration))
    except Exception as exc:
        log.exception("register.ingest_failed", repo=registration.name)
        print_error(f"Ingestion failed: {exc}")
        raise typer.Exit(code=1) from exc

    # Report results
    console.print(
        f"  Files parsed:          {result['files_parsed']}\n"
        f"  Entities created:      {result['entities_created']}\n"
        f"  Relationships created: {result['relationships_created']}\n"
        f"  Duration:              {result['duration_seconds']:.1f}s"
    )
    if result["errors"]:
        count = len(result["errors"])
        console.print(f"  [yellow]Warnings:[/yellow] {count} file(s) had parse errors")

    print_success("Registration complete. Graph populated.")
    sys.exit(0)
