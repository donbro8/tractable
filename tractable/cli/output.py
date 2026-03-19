"""Shared Rich formatting helpers for the tractable CLI."""

from __future__ import annotations

from rich.console import Console

console: Console = Console()
err_console: Console = Console(stderr=True)


def print_success(msg: str) -> None:
    """Print a success message in green."""
    console.print(f"[green]✓[/green] {msg}")


def print_error(msg: str) -> None:
    """Print an error message in red to stderr."""
    err_console.print(f"[bold red]Error:[/bold red] {msg}")
