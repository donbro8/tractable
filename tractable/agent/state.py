"""AgentWorkflowState — TypedDict schema for the LangGraph agent workflow.

TASK-2.3.1: State is passed between all four nodes and updated by each.
"""

from __future__ import annotations

from typing import Any, TypedDict

from tractable.types.enums import TaskPhase


class AgentWorkflowState(TypedDict):
    """Mutable state threaded through every LangGraph workflow node.

    LangGraph merges the dict returned by each node into this state.
    Fields not returned by a node retain their previous value.
    """

    agent_id: str
    task_id: str
    task_description: str
    phase: TaskPhase
    plan: list[str]
    files_changed: list[str]
    test_results: dict[str, Any]
    pr_url: str | None
    error: str | None
    token_count: int
    messages: list[dict[str, Any]]
    resume_from: str | None  # Set by resume_task() when restoring from checkpoint
