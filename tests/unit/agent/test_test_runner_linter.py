"""Unit tests for test_runner.py and linter.py (TASK-2.4.4).

Acceptance criteria covered:
1. run_tests exit 0 → ToolResult(success=True, output["passed"]==True)
2. run_tests exit 1 → ToolResult(success=True, output["passed"]==False), stdout present
3. run_tests TimeoutExpired → RecoverableError
4. run_lint fix=False, 3 violation lines → output["violations"] has 3 items
5. run_lint fix=True → --fix passed to ruff subprocess
6. pyright strict clean (verified via CLI, not here)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tractable.agent.tools.linter import LinterTool
from tractable.agent.tools.test_runner import TestRunnerTool
from tractable.errors import RecoverableError

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_test_runner(tmp_path: Path) -> TestRunnerTool:
    return TestRunnerTool(
        working_dir=tmp_path,
        agent_id="agent-test",
        task_id="task-test",
        repo="test/repo",
    )


def _make_linter(tmp_path: Path) -> LinterTool:
    return LinterTool(
        working_dir=tmp_path,
        agent_id="agent-test",
        task_id="task-test",
        repo="test/repo",
    )


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
    m = MagicMock(spec=subprocess.CompletedProcess)
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


# ── AC-1: exit 0 → passed=True ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_tests_exit_0_returns_passed_true(tmp_path: Path) -> None:
    tool = _make_test_runner(tmp_path)

    with patch(
        "tractable.agent.tools.test_runner.subprocess.run",
        return_value=_completed(0, stdout="3 passed"),
    ):
        result = await tool.invoke({"operation": "run_tests", "test_command": "pytest tests/"})

    assert result.success is True
    assert isinstance(result.output, dict)
    assert result.output["passed"] is True
    assert result.output["exit_code"] == 0


# ── AC-2: exit 1 → passed=False, stdout preserved ────────────────────────────


@pytest.mark.asyncio
async def test_run_tests_exit_1_returns_passed_false_with_output(tmp_path: Path) -> None:
    tool = _make_test_runner(tmp_path)
    failure_output = "FAILED tests/test_foo.py::test_bar"

    with patch(
        "tractable.agent.tools.test_runner.subprocess.run",
        return_value=_completed(1, stdout=failure_output),
    ):
        result = await tool.invoke({"operation": "run_tests", "test_command": "pytest tests/"})

    assert result.success is True
    assert result.output["passed"] is False
    assert result.output["exit_code"] == 1
    assert failure_output in result.output["stdout"]


# ── AC-3: TimeoutExpired → RecoverableError ───────────────────────────────────


@pytest.mark.asyncio
async def test_run_tests_timeout_raises_recoverable_error(tmp_path: Path) -> None:
    tool = _make_test_runner(tmp_path)

    with (
        patch(
            "tractable.agent.tools.test_runner.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="pytest", timeout=5),
        ),
        pytest.raises(RecoverableError, match="timed out"),
    ):
        await tool.invoke(
            {
                "operation": "run_tests",
                "test_command": "pytest tests/",
                "timeout_seconds": 5,
            }
        )


# ── AC-4: run_lint fix=False returns violation lines ─────────────────────────


@pytest.mark.asyncio
async def test_run_lint_returns_violation_lines(tmp_path: Path) -> None:
    tool = _make_linter(tmp_path)
    violation_output = (
        "tractable/foo.py:1:1: E501 line too long\n"
        "tractable/foo.py:2:1: F401 unused import\n"
        "tractable/bar.py:5:3: E302 expected 2 blank lines\n"
    )

    ruff_completed = _completed(1, stdout=violation_output)
    pyright_completed = _completed(0, stdout="0 errors, 0 warnings")

    with patch(
        "tractable.agent.tools.linter.subprocess.run",
        side_effect=[ruff_completed, pyright_completed],
    ):
        result = await tool.invoke({"operation": "run_lint", "fix": False})

    assert result.success is True
    assert len(result.output["violations"]) == 3


# ── AC-5: run_lint fix=True passes --fix to ruff ─────────────────────────────


@pytest.mark.asyncio
async def test_run_lint_fix_true_passes_fix_flag(tmp_path: Path) -> None:
    tool = _make_linter(tmp_path)

    ruff_completed = _completed(0, stdout="Fixed 2 errors in 1 file.\n")
    pyright_completed = _completed(0, stdout="0 errors")

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        calls.append(list(cmd))
        if "ruff" in cmd:
            return ruff_completed
        return pyright_completed

    with patch("tractable.agent.tools.linter.subprocess.run", side_effect=fake_run):
        result = await tool.invoke({"operation": "run_lint", "fix": True})

    ruff_call = next(c for c in calls if "ruff" in c)
    assert "--fix" in ruff_call
    assert result.success is True
    assert result.output.get("fixed_count") == 2


# ── Stdout/stderr truncation ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_tests_stdout_truncated_to_4000_chars(tmp_path: Path) -> None:
    tool = _make_test_runner(tmp_path)
    long_stdout = "x" * 8000

    with patch(
        "tractable.agent.tools.test_runner.subprocess.run",
        return_value=_completed(0, stdout=long_stdout),
    ):
        result = await tool.invoke({"operation": "run_tests", "test_command": "pytest"})

    assert len(result.output["stdout"]) == 4000


@pytest.mark.asyncio
async def test_run_tests_stderr_truncated_to_2000_chars(tmp_path: Path) -> None:
    tool = _make_test_runner(tmp_path)
    long_stderr = "e" * 5000

    with patch(
        "tractable.agent.tools.test_runner.subprocess.run",
        return_value=_completed(1, stderr=long_stderr),
    ):
        result = await tool.invoke({"operation": "run_tests", "test_command": "pytest"})

    assert len(result.output["stderr"]) == 2000


# ── Unknown operation ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_test_runner_unknown_operation_returns_failure(tmp_path: Path) -> None:
    tool = _make_test_runner(tmp_path)
    result = await tool.invoke({"operation": "nonexistent"})
    assert result.success is False


@pytest.mark.asyncio
async def test_linter_unknown_operation_returns_failure(tmp_path: Path) -> None:
    tool = _make_linter(tmp_path)
    result = await tool.invoke({"operation": "nonexistent"})
    assert result.success is False
