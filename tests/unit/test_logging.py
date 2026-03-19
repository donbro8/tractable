"""Unit tests for tractable/logging.py.

Covers:
- JSON output format (production env)
- Console output format (development env)
- Context binding (all fields)
- Context clearing
- Partial context (only some fields provided)
"""

from __future__ import annotations

import io
import json

import pytest
import structlog
import structlog.contextvars

from tractable.logging import bind_context, clear_context, configure_logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_log_output(env: str) -> tuple[io.StringIO, structlog.stdlib.BoundLogger]:
    """Configure logging for *env*, return (buffer, logger) pair."""
    buf = io.StringIO()

    # Reconfigure structlog to write into our buffer instead of stdout.
    if env == "production":
        processors: list[object] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ]

    structlog.configure(
        processors=processors,  # type: ignore[arg-type]
        wrapper_class=structlog.make_filtering_bound_logger(10),  # DEBUG
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=buf),
        cache_logger_on_first_use=False,
    )

    logger = structlog.get_logger()
    return buf, logger  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# AC-1: import check (tested implicitly; if it fails every test fails)
# ---------------------------------------------------------------------------


def test_imports() -> None:
    """configure_logging, bind_context, clear_context are importable."""
    from tractable.logging import bind_context, clear_context, configure_logging  # noqa: F401


# ---------------------------------------------------------------------------
# AC-2: JSON output contains required fields
# ---------------------------------------------------------------------------


class TestProductionJsonOutput:
    def setup_method(self) -> None:
        clear_context()

    def teardown_method(self) -> None:
        clear_context()

    def test_json_output_contains_all_required_fields(self) -> None:
        buf, logger = _capture_log_output("production")
        bind_context(agent_id="a1", task_id="t1", repo="my-repo")
        # In structlog the first positional arg IS the event field.
        logger.info("test_event")

        output = buf.getvalue().strip()
        assert output, "Expected log output but got empty string"
        record = json.loads(output)

        assert record["agent_id"] == "a1"
        assert record["task_id"] == "t1"
        assert record["repo"] == "my-repo"
        assert record["event"] == "test_event"
        assert record["level"] == "info"

    def test_json_output_is_single_line(self) -> None:
        buf, logger = _capture_log_output("production")
        bind_context(agent_id="a1", task_id="t1", repo="my-repo")
        logger.info("e")

        lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        assert len(lines) == 1

    def test_configure_logging_production_does_not_raise(self) -> None:
        configure_logging(env="production")


# ---------------------------------------------------------------------------
# AC-3: Console output is NOT JSON
# ---------------------------------------------------------------------------


class TestDevelopmentConsoleOutput:
    def setup_method(self) -> None:
        clear_context()

    def teardown_method(self) -> None:
        clear_context()

    def test_console_output_does_not_start_with_brace(self) -> None:
        buf, logger = _capture_log_output("development")
        logger.info("dev_event")

        output = buf.getvalue().strip()
        assert output, "Expected log output but got empty string"
        assert not output.startswith("{"), (
            f"Expected non-JSON output in development mode, got: {output!r}"
        )

    def test_configure_logging_development_does_not_raise(self) -> None:
        configure_logging(env="development")


# ---------------------------------------------------------------------------
# AC-4: clear_context removes bound vars
# ---------------------------------------------------------------------------


class TestContextClearing:
    def setup_method(self) -> None:
        clear_context()

    def teardown_method(self) -> None:
        clear_context()

    def test_clear_context_removes_agent_id_task_id_repo(self) -> None:
        buf, logger = _capture_log_output("production")
        bind_context(agent_id="a1", task_id="t1", repo="my-repo")
        clear_context()
        logger.info("after_clear")

        output = buf.getvalue().strip()
        assert output
        record = json.loads(output)

        assert "agent_id" not in record
        assert "task_id" not in record
        assert "repo" not in record

    def test_clear_context_then_rebind_works(self) -> None:
        buf, logger = _capture_log_output("production")
        bind_context(agent_id="old", task_id="old", repo="old-repo")
        clear_context()
        bind_context(agent_id="new-agent")
        logger.info("rebind")

        output = buf.getvalue().strip()
        record = json.loads(output)
        assert record["agent_id"] == "new-agent"
        assert "task_id" not in record
        assert "repo" not in record


# ---------------------------------------------------------------------------
# Additional: partial context (only some fields provided)
# ---------------------------------------------------------------------------


class TestPartialContext:
    def setup_method(self) -> None:
        clear_context()

    def teardown_method(self) -> None:
        clear_context()

    def test_only_agent_id_bound(self) -> None:
        buf, logger = _capture_log_output("production")
        bind_context(agent_id="agent-only")
        logger.info("partial")

        record = json.loads(buf.getvalue().strip())
        assert record["agent_id"] == "agent-only"
        assert "task_id" not in record
        assert "repo" not in record

    def test_only_repo_bound(self) -> None:
        buf, logger = _capture_log_output("production")
        bind_context(repo="my-repo-only")
        logger.info("partial_repo")

        record = json.loads(buf.getvalue().strip())
        assert record["repo"] == "my-repo-only"
        assert "agent_id" not in record
        assert "task_id" not in record

    def test_no_context_bound_omits_context_fields(self) -> None:
        buf, logger = _capture_log_output("production")
        logger.info("no_context")

        record = json.loads(buf.getvalue().strip())
        assert "agent_id" not in record
        assert "task_id" not in record
        assert "repo" not in record
        assert record["event"] == "no_context"

    def test_none_values_are_omitted(self) -> None:
        """bind_context(agent_id=None) should not add agent_id to the log."""
        buf, logger = _capture_log_output("production")
        bind_context(agent_id=None, task_id="t99", repo=None)
        logger.info("none_values")

        record = json.loads(buf.getvalue().strip())
        assert "agent_id" not in record
        assert record["task_id"] == "t99"
        assert "repo" not in record
