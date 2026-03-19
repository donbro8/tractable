"""test_runner tool — run the project's test suite and report results.

TASK-2.4.4: Implements the ``test_runner`` tool satisfying the ``Tool``
Protocol (tech-spec.py §2.6).  A non-zero exit code is surfaced as
``output["passed"] = False`` rather than an exception — the agent reads
failure output and re-plans.  A timeout raises ``RecoverableError`` so the
agent can retry with a shorter suite or a longer budget.

Sources:
- tech-spec.py §2.6 — Tool Protocol
- CLAUDE.md        — Error taxonomy
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import structlog

from tractable.errors import RecoverableError
from tractable.protocols.tool import ToolResult

_log = structlog.get_logger()

_STDOUT_TAIL = 4000
_STDERR_TAIL = 2000


class TestRunnerTool:
    """Run the project test suite and return structured pass/fail output.

    Parameters
    ----------
    working_dir:
        Absolute path to the repository root (subprocess cwd).
    agent_id:
        Identifier of the running agent (included in every log entry).
    task_id:
        Identifier of the current task (included in every log entry).
    repo:
        Human-readable repository name (included in every log entry).
    """

    def __init__(
        self,
        working_dir: Path,
        agent_id: str,
        task_id: str,
        repo: str,
    ) -> None:
        self._working_dir = working_dir.resolve()
        self._agent_id = agent_id
        self._task_id = task_id
        self._repo = repo

    # ── Tool Protocol ────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "test_runner"

    @property
    def description(self) -> str:
        return (
            "Run the project test suite. Returns exit_code, stdout, stderr, "
            "and a boolean 'passed' flag. Non-zero exit is NOT an exception — "
            "read the output and re-plan."
        )

    async def invoke(self, params: dict[str, Any]) -> ToolResult:
        """Dispatch to the appropriate operation.

        Expected ``params`` keys:

        - ``operation``: ``"run_tests"``
        - ``test_command``: shell command string to execute
        - ``timeout_seconds``: max seconds before RecoverableError (default 120)
        """
        operation: str = params.get("operation", "")

        if operation == "run_tests":
            return await self._run_tests(params)

        return ToolResult(success=False, error=f"Unknown operation: {operation!r}")

    # ── Operations ───────────────────────────────────────────────────────────

    async def _run_tests(self, params: dict[str, Any]) -> ToolResult:
        test_command: str = params.get("test_command", "")
        timeout_seconds: int = int(params.get("timeout_seconds", 120))

        start_ms = time.monotonic()
        try:
            result = subprocess.run(
                test_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=self._working_dir,
            )
        except subprocess.TimeoutExpired as exc:
            raise RecoverableError(
                f"Test runner timed out after {timeout_seconds}s"
            ) from exc

        duration_ms = int((time.monotonic() - start_ms) * 1000)

        stdout = (
            result.stdout[-_STDOUT_TAIL:] if len(result.stdout) > _STDOUT_TAIL else result.stdout
        )
        stderr = (
            result.stderr[-_STDERR_TAIL:] if len(result.stderr) > _STDERR_TAIL else result.stderr
        )

        _log.info(
            "tests_run",
            level="info",
            agent_id=self._agent_id,
            task_id=self._task_id,
            repo=self._repo,
            exit_code=result.returncode,
            duration_ms=duration_ms,
        )

        return ToolResult(
            success=True,
            output={
                "exit_code": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "passed": result.returncode == 0,
            },
        )
