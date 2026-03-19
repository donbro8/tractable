"""tractable CLI entry point.

Usage::

    tractable register <config.py>   # onboard a repository
    tractable status                 # show registered agents
"""

from __future__ import annotations

import typer

from tractable.cli.commands.register import register_app
from tractable.cli.commands.status import status_app

app: typer.Typer = typer.Typer(
    name="tractable",
    no_args_is_help=True,
    help="Tractable — autonomous multi-agent coding framework.",
)

app.add_typer(register_app, name="register")
app.add_typer(status_app, name="status")


def main() -> None:
    """CLI entry point registered in pyproject.toml."""
    app()
