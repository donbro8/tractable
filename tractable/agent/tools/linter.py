"""linter tool — run ruff and pyright over the managed repository.

TASK-2.4.4: Implements the ``linter`` tool satisfying the ``Tool`` Protocol
(tech-spec.py §2.6).  The agent calls this during the REVIEWING node to gate
PR creation on ``requires_lint_pass`` / ``requires_tests_pass`` governance
flags.

Runs two sub-commands:
1. ``uv run ruff check [--fix] tractable/`` — returns violation lines and
   optional ``fixed_count`` when ``fix=True``.
2. ``uv run pyright tractable/`` — returns ``pyright_errors`` lines.

Sources:
- tech-spec.py §2.6 — Tool Protocol
- CLAUDE.md        — Governance enforcement points
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Any

import structlog

from tractable.protocols.tool import ToolResult

_log = structlog.get_logger()

_FIXED_RE = re.compile(r"Fixed (\d+) error", re.IGNORECASE)


class LinterTool:
    """Run ruff and pyright over the repository and return structured output.

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
        return "linter"

    @property
    def description(self) -> str:
        return (
            "Run ruff (style/lint) and pyright (type-check) over tractable/. "
            "Returns violations and pyright errors so the agent can gate PR "
            "creation on lint/type cleanliness."
        )

    async def invoke(self, params: dict[str, Any]) -> ToolResult:
        """Dispatch to the appropriate operation.

        Expected ``params`` keys:

        - ``operation``: ``"run_lint"``
        - ``fix``: bool — whether to pass ``--fix`` to ruff (default False)
        """
        operation: str = params.get("operation", "")

        if operation == "run_lint":
            return await self._run_lint(params)

        return ToolResult(success=False, error=f"Unknown operation: {operation!r}")

    # ── Operations ───────────────────────────────────────────────────────────

    async def _run_lint(self, params: dict[str, Any]) -> ToolResult:
        fix: bool = bool(params.get("fix", False))

        start_ms = time.monotonic()

        ruff_cmd = ["uv", "run", "ruff", "check"]
        if fix:
            ruff_cmd.append("--fix")
        ruff_cmd.append("tractable/")

        ruff_result = subprocess.run(
            ruff_cmd,
            capture_output=True,
            text=True,
            cwd=self._working_dir,
        )

        pyright_result = subprocess.run(
            ["uv", "run", "pyright", "tractable/"],
            capture_output=True,
            text=True,
            cwd=self._working_dir,
        )

        duration_ms = int((time.monotonic() - start_ms) * 1000)

        ruff_output = ruff_result.stdout + ruff_result.stderr
        violations = [line for line in ruff_output.splitlines() if line.strip()]

        pyright_output = pyright_result.stdout + pyright_result.stderr
        pyright_errors = [line for line in pyright_output.splitlines() if line.strip()]

        fixed_count = 0
        if fix:
            match = _FIXED_RE.search(ruff_output)
            if match:
                fixed_count = int(match.group(1))

        exit_code = ruff_result.returncode

        _log.info(
            "lint_run",
            level="info",
            agent_id=self._agent_id,
            task_id=self._task_id,
            repo=self._repo,
            exit_code=exit_code,
            duration_ms=duration_ms,
        )

        output: dict[str, Any] = {
            "exit_code": exit_code,
            "violations": violations,
            "pyright_errors": pyright_errors,
        }
        if fix:
            output["fixed_count"] = fixed_count

        return ToolResult(success=True, output=output)
