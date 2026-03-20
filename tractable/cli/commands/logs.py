"""tractable logs — stream and query the audit log.

Usage:
    tractable logs [--agent <id>] [--task <id>] [--follow]
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from datetime import datetime

import structlog
import typer
from rich.live import Live

from tractable.cli.output import console
from tractable.errors import FatalError
from tractable.types.agent import AuditEntry

log = structlog.get_logger(__name__)

logs_app = typer.Typer(help="Query and stream the agent audit log.")

_DEFAULT_LIMIT = 50
_POLL_INTERVAL_SECONDS = 2


# ── Async helpers (mockable in tests) ────────────────────────────────────────


async def _fetch_log(
    agent_id: str | None,
    task_id: str | None,
    since: datetime | None,
    limit: int,
) -> Sequence[AuditEntry]:
    """Query the audit log from the state store."""
    from tractable.state.store import PostgreSQLAgentStateStore

    store = PostgreSQLAgentStateStore.from_env()
    return await store.get_audit_log(
        agent_id=agent_id,
        task_id=task_id,
        since=since,
        limit=limit,
    )


# ── Commands ──────────────────────────────────────────────────────────────────


@logs_app.callback(invoke_without_command=True)
def logs_cmd(
    ctx: typer.Context,
    agent: str | None = typer.Option(None, "--agent", help="Filter by agent ID."),
    task: str | None = typer.Option(None, "--task", help="Filter by task ID."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Poll for new entries every 2 s."),
) -> None:
    """Print audit log entries as JSON lines.

    Without filters, returns the last 50 entries across all agents.
    Use --follow to tail new entries continuously.
    """
    if ctx.invoked_subcommand is not None:
        return

    if not follow:
        # One-shot fetch.
        try:
            entries = asyncio.run(_fetch_log(agent, task, since=None, limit=_DEFAULT_LIMIT))
        except FatalError:
            raise
        except Exception as exc:
            log.exception("logs.fetch_failed")
            raise FatalError(f"Could not fetch audit log: {exc}") from exc

        for entry in entries:
            print(entry.model_dump_json())
        return

    # Follow mode: poll every 2 s for new entries.
    seen_since: datetime | None = None
    try:
        with Live(console=console, auto_refresh=False):
            while True:
                try:
                    entries = asyncio.run(
                        _fetch_log(agent, task, since=seen_since, limit=_DEFAULT_LIMIT)
                    )
                except FatalError:
                    raise
                except Exception as exc:
                    log.exception("logs.poll_failed")
                    raise FatalError(f"Could not fetch audit log: {exc}") from exc

                for entry in entries:
                    print(entry.model_dump_json())

                if entries:
                    # get_audit_log returns DESC order; newest is first.
                    seen_since = entries[0].timestamp

                time.sleep(_POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        pass
