"""tractable task — submit and manage tasks.

Subcommands:
    submit "<description>" --repo <name>   — submit a task to a registered repo
"""

from __future__ import annotations

import asyncio
import uuid

import structlog
import typer

from tractable.errors import FatalError

log = structlog.get_logger(__name__)

task_app = typer.Typer(help="Submit and manage tasks.")


# ── Async helpers (mockable in tests) ────────────────────────────────────────


async def _submit_task_async(description: str, repo_name: str) -> str:
    """Validate repo is registered and create a task. Returns the task_id.

    Raises ``FatalError`` if ``repo_name`` is not a registered repository.
    """
    from tractable.state.store import PostgreSQLAgentStateStore

    store = PostgreSQLAgentStateStore.from_env()
    agents = await store.list_agents()
    if not any(a.repo == repo_name for a in agents):
        raise FatalError(f"Repo {repo_name} not found.")

    task_id = str(uuid.uuid4())
    log.info(
        "task_submitted",
        task_id=task_id,
        repo=repo_name,
        description=description,
    )
    return task_id


# ── Commands ──────────────────────────────────────────────────────────────────


@task_app.command("submit")
def task_submit(
    description: str = typer.Argument(..., help="Natural-language task description."),
    repo: str = typer.Option(..., "--repo", help="Registered repo name to assign the task to."),
) -> None:
    """Submit a task to a registered repository."""
    try:
        task_id = asyncio.run(_submit_task_async(description, repo))
    except FatalError:
        raise
    except Exception as exc:
        log.exception("task_submit.failed", repo=repo)
        raise FatalError(f"Could not submit task: {exc}") from exc

    print(task_id)
