"""Integration test: EXECUTING node pauses on sensitive path write (TASK-3.2.4).

Acceptance criterion AC-4:
    When a ``code_editor`` write targets a path matching a ``SensitivePathRule``,
    the EXECUTING node returns ``phase=TaskPhase.REVIEWING`` and ``error``
    contains the matched rule pattern.

This test is self-contained (no external services required) and verifies the
integration between ``CodeEditorTool``, ``GovernanceError``, and the EXECUTING
node's catch logic by injecting a real ``CodeEditorTool`` with a sensitive-path
governance policy.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tractable.agent.nodes.execute import make_executing_node
from tractable.agent.state import AgentWorkflowState
from tractable.errors import GovernanceError
from tractable.protocols.tool import ToolResult
from tractable.types.enums import TaskPhase

# ── Helpers ────────────────────────────────────────────────────────────────


class _SensitivePathCodeEditor:
    """Code editor stub that raises GovernanceError for any write attempt.

    Simulates a ``CodeEditorTool`` that has already resolved the target path
    as matching a sensitive-path rule.
    """

    name = "code_editor"
    description = "stub"

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        raise GovernanceError(
            "Sensitive path blocked: /repo/db/migrations/003_create_users.sql "
            "matches rule '**/migrations/**' "
            "(Database migrations require DBA review)"
        )


def _make_state(
    plan: list[str] | None = None,
    files_changed: list[str] | None = None,
) -> AgentWorkflowState:
    state = AgentWorkflowState(
        agent_id="agent-test",
        task_id="task-test",
        task_description="test task",
        phase=TaskPhase.EXECUTING,
        plan=plan or [],
        files_changed=files_changed or [],
        test_results={},
        pr_url=None,
        error=None,
        token_count=0,
        current_model="claude-sonnet-4-6",
        messages=[],
        resume_from=None,
    )
    return state


def _stub_state_store() -> AsyncMock:
    store = AsyncMock()
    store.save_checkpoint = AsyncMock()
    return store


# ── Test ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_executing_node_pauses_on_sensitive_path() -> None:
    """AC-4: EXECUTING node sets phase=REVIEWING + error contains matched rule pattern."""
    code_editor = _SensitivePathCodeEditor()
    state = _make_state(
        plan=["write db/migrations/003_create_users.sql"],
        files_changed=[],
    )

    node = make_executing_node(
        {"code_editor": code_editor},  # type: ignore[dict-item]
        _stub_state_store(),
    )
    result = await node(state)

    assert result["phase"] == TaskPhase.REVIEWING, (
        f"Expected REVIEWING but got {result['phase']!r}; error={result.get('error')!r}"
    )
    assert result.get("error") is not None, "Expected error to be set"
    assert "migrations/**" in str(result["error"]), (
        f"Expected rule pattern in error; got: {result['error']!r}"
    )
