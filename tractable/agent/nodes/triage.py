# pyright: reportUnknownMemberType=false, reportArgumentType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false
"""PIPELINE_TRIAGE sub-workflow — classify and handle CI failures.

TASK-2.4.5: Separate LangGraph StateGraph invoked from the REVIEWING node when
``pipeline_watcher`` reports ``any_failed=True``.

Workflow
--------
1. Classify the failure via the injected ``classify_fn`` (LLM call or mock):
   - ``"flaky"``        — random infra issue; re-trigger the failed checks
   - ``"agent_caused"`` — agent's code broke a test; return to EXECUTING with
                          a ``"CI failure reason: …"`` message in state
   - ``"environment"``  — missing dependency / wrong Python version; raise
                          ``GovernanceError`` so the COORDINATING node can post
                          a PR comment and halt for human review

Sources:
- tech-spec.py §2.6 — Tool Protocol
- tech-spec.py §2.1 — GitProvider Protocol
- CLAUDE.md        — Error taxonomy, structured logging
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypedDict

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from tractable.errors import GovernanceError
from tractable.types.enums import TaskPhase

if TYPE_CHECKING:
    from tractable.protocols.git_provider import GitProvider

_log = structlog.get_logger()

# Type alias for the LLM classification function.
# Accepts the failure log text; returns one of "flaky", "agent_caused", "environment".
ClassifyFn = Callable[[str], Awaitable[str]]


# ── State ────────────────────────────────────────────────────────────────────


class TriageState(TypedDict):
    """State for the PIPELINE_TRIAGE sub-workflow."""

    agent_id: str
    task_id: str
    repo_id: str
    pr_number: int
    failure_logs: str
    classification: str | None
    messages: list[dict[str, Any]]
    phase: TaskPhase


# ── Default LLM classification function ──────────────────────────────────────


async def anthropic_classify_fn(failure_logs: str) -> str:
    """Classify a CI failure using the Anthropic API.

    This is the default ``classify_fn`` for :func:`build_triage_graph`.

    Returns one of ``"flaky"``, ``"agent_caused"``, or ``"environment"``.
    Defaults to ``"environment"`` when the response cannot be parsed.
    """
    import anthropic  # local import avoids hard dep at module load time

    client = anthropic.AsyncAnthropic()
    message = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify this CI failure as exactly one of three categories:\n"
                    '- "flaky" — random test infrastructure issue, not caused by code changes\n'
                    '- "agent_caused" — the code change broke a test\n'
                    '- "environment" — missing dependency, wrong Python version, infra config\n\n'
                    f"Failure log:\n{failure_logs[:2000]}\n\n"
                    "Reply with just the single classification word."
                ),
            }
        ],
    )
    content = message.content
    if content and isinstance(content[0], anthropic.types.TextBlock):
        text = content[0].text.strip().lower()
        if "flaky" in text:
            return "flaky"
        if "agent" in text:
            return "agent_caused"
    return "environment"


# ── Sub-workflow factory ──────────────────────────────────────────────────────


def build_triage_graph(
    classify_fn: ClassifyFn,
    git_provider: GitProvider,
) -> CompiledStateGraph:
    """Construct and compile the PIPELINE_TRIAGE LangGraph sub-workflow.

    Parameters
    ----------
    classify_fn:
        Async callable that accepts the failure log text and returns one of
        ``"flaky"``, ``"agent_caused"``, or ``"environment"``.  Inject a mock
        in unit tests to avoid real LLM calls.
    git_provider:
        GitProvider used to re-trigger failed checks on the ``"flaky"`` path.

    Returns
    -------
    CompiledStateGraph
        Ready-to-invoke LangGraph sub-workflow over ``TriageState``.
    """

    async def triage_node(state: TriageState) -> dict[str, Any]:
        agent_id = state["agent_id"]
        task_id = state["task_id"]
        failure_logs = state["failure_logs"]

        classification = await classify_fn(failure_logs)

        _log.info(
            "ci_failure_classified",
            level="info",
            agent_id=agent_id,
            task_id=task_id,
            repo=state["repo_id"],
            classification=classification,
        )

        if classification == "agent_caused":
            new_message: dict[str, Any] = {
                "role": "user",
                "content": f"CI failure reason: {failure_logs[:500]}",
            }
            return {
                "classification": classification,
                "phase": TaskPhase.EXECUTING,
                "messages": [*state["messages"], new_message],
            }

        if classification == "flaky":
            await git_provider.rerun_failed_checks(state["repo_id"], state["pr_number"])
            _log.info(
                "ci_checks_rerun",
                level="info",
                agent_id=agent_id,
                task_id=task_id,
                repo=state["repo_id"],
            )
            return {
                "classification": classification,
                "phase": state["phase"],
            }

        # "environment" or any unrecognised classification → GovernanceError
        raise GovernanceError(
            f"CI failure classified as environment issue — human review required. "
            f"Log excerpt: {failure_logs[:500]}"
        )

    builder: StateGraph = StateGraph(TriageState)
    builder.add_node("triage", triage_node)
    builder.add_edge(START, "triage")
    builder.add_edge("triage", END)

    return builder.compile()
