"""tractable agent — manage agent contexts and pinned instructions.

Subcommands:
    list            — display all registered agents
    context <id>    — print assembled system prompt for an agent
    edit <id>       — edit pinned instructions in $EDITOR
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from contextlib import suppress
from typing import Any

import structlog
import typer
from rich.table import Table

from tractable.cli.output import console
from tractable.errors import FatalError, RecoverableError

log = structlog.get_logger(__name__)

agent_app = typer.Typer(help="Manage agent contexts and pinned instructions.")


# ── Async helpers (mockable in tests) ────────────────────────────────────────


async def _list_agents_data() -> list[dict[str, Any]]:
    """Fetch all agent contexts from the state store for display."""
    from tractable.state.store import PostgreSQLAgentStateStore

    store = PostgreSQLAgentStateStore.from_env()
    agents = await store.list_agents()
    return [
        {
            "agent_id": a.agent_id,
            "repo": a.repo or "—",
            "status": str(a.user_overrides.get("status", "—")),
            "last_active": str(a.user_overrides.get("last_active", "—")),
            "current_task_id": str(a.user_overrides.get("current_task_id", "—")),
        }
        for a in agents
    ]


async def _get_context_text(agent_id: str) -> str:
    """Return the assembled context string for display.

    Reads the stored system_prompt and appends any current pinned instructions.
    """
    from tractable.state.store import PostgreSQLAgentStateStore

    store = PostgreSQLAgentStateStore.from_env()
    try:
        ctx = await store.get_agent_context(agent_id)
    except RecoverableError as exc:
        raise FatalError(f"Agent {agent_id} not found.") from exc

    parts: list[str] = [ctx.system_prompt]
    for instr in ctx.pinned_instructions:
        parts.append(f"[pinned] {instr}")
    return "\n".join(parts)


async def _edit_pinned(agent_id: str, editor: str) -> None:
    """Open pinned_instructions in an editor and save changes."""
    from tractable.state.store import PostgreSQLAgentStateStore

    store = PostgreSQLAgentStateStore.from_env()
    try:
        ctx = await store.get_agent_context(agent_id)
    except RecoverableError as exc:
        raise FatalError(f"Agent {agent_id} not found.") from exc

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp_path = tmp.name
        json.dump(ctx.pinned_instructions, tmp, indent=2)

    try:
        subprocess.run([editor, tmp_path], check=True)  # noqa: S603
        with open(tmp_path, encoding="utf-8") as f:
            updated: list[str] = json.load(f)
        ctx.pinned_instructions = updated
        await store.save_agent_context(agent_id, ctx)
    finally:
        with suppress(OSError):
            os.unlink(tmp_path)


# ── Commands ──────────────────────────────────────────────────────────────────


@agent_app.command("list")
def agent_list() -> None:
    """Display all registered agents in a table."""
    try:
        rows = asyncio.run(_list_agents_data())
    except FatalError:
        raise
    except Exception as exc:
        log.exception("agent_list.failed")
        raise FatalError(f"Could not fetch agents: {exc}") from exc

    if not rows:
        console.print("No agents registered. Run `tractable register <config.py>` to start.")
        return

    table = Table(
        "Agent ID",
        "Repo",
        "Status",
        "Last Active",
        "Current Task ID",
        title="Registered Agents",
    )
    for row in rows:
        table.add_row(
            row["agent_id"],
            row["repo"],
            row["status"],
            row["last_active"],
            row["current_task_id"],
        )
    console.print(table)


@agent_app.command("context")
def agent_context(
    agent_id: str = typer.Argument(..., help="Agent ID to inspect."),
) -> None:
    """Print the assembled system prompt for an agent."""
    try:
        text = asyncio.run(_get_context_text(agent_id))
    except FatalError:
        raise
    except Exception as exc:
        log.exception("agent_context.failed", agent_id=agent_id)
        raise FatalError(f"Could not load agent context: {exc}") from exc

    print(text)


@agent_app.command("edit")
def agent_edit(
    agent_id: str = typer.Argument(..., help="Agent ID whose pinned instructions to edit."),
) -> None:
    """Edit pinned instructions for an agent in $EDITOR."""
    editor = os.environ.get("EDITOR", "")

    if not editor:
        # Fallback: print current instructions and accept JSON from stdin.
        from tractable.state.store import PostgreSQLAgentStateStore

        async def _get_pinned() -> list[str]:
            store = PostgreSQLAgentStateStore.from_env()
            try:
                ctx = await store.get_agent_context(agent_id)
            except RecoverableError as exc:
                raise FatalError(f"Agent {agent_id} not found.") from exc
            return list(ctx.pinned_instructions)

        pinned = asyncio.run(_get_pinned())
        console.print("Current pinned instructions (JSON):")
        console.print(json.dumps(pinned, indent=2))
        console.print("\nEnter updated JSON (press Ctrl+D when done):")
        try:
            raw = sys.stdin.read().strip()
            updated: list[str] = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise FatalError(f"Invalid JSON: {exc}") from exc

        async def _save_pinned(new_pinned: list[str]) -> None:
            store = PostgreSQLAgentStateStore.from_env()
            try:
                ctx = await store.get_agent_context(agent_id)
            except RecoverableError as ex:
                raise FatalError(f"Agent {agent_id} not found.") from ex
            ctx.pinned_instructions = new_pinned
            await store.save_agent_context(agent_id, ctx)

        try:
            asyncio.run(_save_pinned(updated))
        except FatalError:
            raise
        except Exception as exc:
            raise FatalError(f"Could not save pinned instructions: {exc}") from exc
        console.print("[green]Pinned instructions updated.[/green]")
        return

    try:
        asyncio.run(_edit_pinned(agent_id, editor))
    except FatalError:
        raise
    except Exception as exc:
        log.exception("agent_edit.failed", agent_id=agent_id)
        raise FatalError(f"Could not edit pinned instructions: {exc}") from exc

    console.print("[green]Pinned instructions updated.[/green]")
