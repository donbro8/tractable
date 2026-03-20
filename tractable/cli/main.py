"""tractable CLI entry point.

Usage::

    tractable register <config.py>   # onboard a repository
    tractable status                 # show registered agents
    tractable agent list             # list agents with status
    tractable agent context <id>     # print assembled system prompt
    tractable agent edit <id>        # edit pinned instructions
    tractable task submit <desc> --repo <name>   # submit a task
    tractable logs [--agent <id>] [--task <id>] [--follow]
"""

from __future__ import annotations

import sys

import typer

from tractable.cli.commands.agent import agent_app
from tractable.cli.commands.logs import logs_app
from tractable.cli.commands.register import register_app
from tractable.cli.commands.status import status_app
from tractable.cli.commands.task import task_app
from tractable.errors import FatalError
from tractable.logging import configure_logging

app: typer.Typer = typer.Typer(
    name="tractable",
    no_args_is_help=True,
    help="Tractable — autonomous multi-agent coding framework.",
)

app.add_typer(register_app, name="register")
app.add_typer(status_app, name="status")
app.add_typer(agent_app, name="agent")
app.add_typer(task_app, name="task")
app.add_typer(logs_app, name="logs")


def main() -> None:
    """CLI entry point registered in pyproject.toml."""
    configure_logging()
    try:
        app()
    except FatalError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
