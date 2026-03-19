"""tractable status — display registered agent contexts from PostgreSQL."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import structlog
import typer
from rich.table import Table

from tractable.cli.output import console, print_error

log = structlog.get_logger(__name__)

status_app = typer.Typer(help="Show status of registered agents.")

_NO_AGENTS_MSG = "No agents registered. Run `tractable register <config.py>` to start."


async def _fetch_contexts() -> list[dict[str, Any]]:
    """Fetch all AgentContextORM rows from PostgreSQL."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from tractable.state.models import AgentContextORM

    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set.")

    engine = create_async_engine(url, pool_pre_ping=True)
    factory: async_sessionmaker[Any] = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with factory() as session:
            rows = (await session.execute(select(AgentContextORM))).scalars().all()
            return [
                {
                    "agent_id": r.agent_id,
                    "base_template": r.base_template,
                    "last_active": str(r.last_active) if r.last_active else "—",
                    "last_known_head_sha": (r.last_known_head_sha or "—")[:12],
                }
                for r in rows
            ]
    finally:
        await engine.dispose()


@status_app.callback(invoke_without_command=True)
def status(ctx: typer.Context) -> None:
    """Display all registered agent contexts."""
    if ctx.invoked_subcommand is not None:
        return

    try:
        contexts = asyncio.run(_fetch_contexts())
    except RuntimeError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        log.exception("status.fetch_failed")
        print_error(f"Could not connect to database: {exc}")
        raise typer.Exit(code=1) from exc

    if not contexts:
        console.print(_NO_AGENTS_MSG)
        return

    table = Table(
        "Agent ID",
        "Repo / Template",
        "Last Active",
        "Head SHA",
        title="Registered Agents",
    )
    for ctx_row in contexts:
        table.add_row(
            ctx_row["agent_id"],
            ctx_row["base_template"] or "—",
            ctx_row["last_active"],
            ctx_row["last_known_head_sha"],
        )

    console.print(table)
