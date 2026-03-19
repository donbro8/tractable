"""tractable CLI entry point.

Usage::

    tractable register <config.py>   # onboard a repository
    tractable status                 # show registered agents
"""

from __future__ import annotations

import sys

import typer

from tractable.cli.commands.register import register_app
from tractable.cli.commands.status import status_app
from tractable.errors import FatalError
from tractable.logging import configure_logging

app: typer.Typer = typer.Typer(
    name="tractable",
    no_args_is_help=True,
    help="Tractable — autonomous multi-agent coding framework.",
)

app.add_typer(register_app, name="register")
app.add_typer(status_app, name="status")


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
