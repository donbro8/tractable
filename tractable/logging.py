"""Structured logging initialization and context binding for Tractable.

Uses ``structlog.contextvars`` rather than plain thread-locals so that bound
context (``agent_id``, ``task_id``, ``repo``) is carried correctly through
``asyncio`` event-loop tasks.  A standard ``threading.local`` would be lost
the moment execution crosses an ``await`` boundary; ``contextvars`` propagates
through ``asyncio.Task`` copies of the context automatically.

Public API
----------
configure_logging(env)
    Call once at process start-up (CLI entry point, FastAPI ``startup`` hook).
bind_context(agent_id, task_id, repo)
    Bind structured fields into the current async context.
clear_context()
    Remove all bound context vars (useful between requests / tasks).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = ("agent_id", "task_id", "repo", "event", "level")

_JSON_PROCESSORS: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.processors.UnicodeDecoder(),
    structlog.processors.JSONRenderer(),
]

_CONSOLE_PROCESSORS: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.TimeStamper(fmt="%H:%M:%S"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.dev.ConsoleRenderer(),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(env: str | None = None) -> None:
    """Initialise structlog globally.

    Parameters
    ----------
    env:
        ``"production"`` — JSON output (one dict per line, to stdout).
        ``"development"`` (default) — coloured console output via
        :class:`structlog.dev.ConsoleRenderer`.

        If *env* is ``None`` the value is read from the ``TRACTABLE_ENV``
        environment variable; if that is also absent, ``"development"`` is
        assumed.
    """
    resolved_env = env if env is not None else os.environ.get("TRACTABLE_ENV", "development")

    processors = _JSON_PROCESSORS if resolved_env == "production" else _CONSOLE_PROCESSORS

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def bind_context(
    agent_id: str | None = None,
    task_id: str | None = None,
    repo: str | None = None,
) -> None:
    """Bind structured fields into the current async context.

    All subsequent ``structlog.get_logger()`` calls in the same
    :mod:`contextvars` context will include these fields automatically.
    Fields whose value is ``None`` are omitted from log output.

    Parameters
    ----------
    agent_id:
        Unique identifier of the agent emitting the log entry.
    task_id:
        Unique identifier of the current task being executed.
    repo:
        Short name or URL of the repository the agent is working on.
    """
    values: dict[str, str] = {}
    if agent_id is not None:
        values["agent_id"] = agent_id
    if task_id is not None:
        values["task_id"] = task_id
    if repo is not None:
        values["repo"] = repo

    structlog.contextvars.bind_contextvars(**values)


def clear_context() -> None:
    """Remove all bound context vars from the current async context.

    A subsequent log call will not include ``agent_id``, ``task_id``, or
    ``repo`` unless ``bind_context`` is called again.
    """
    structlog.contextvars.clear_contextvars()
