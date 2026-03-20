# pyright: reportUnknownMemberType=false, reportArgumentType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false
"""LangGraph four-node agent workflow with checkpoint persistence.

TASK-2.3.1 — Implement the PLANNING → EXECUTING → REVIEWING → COORDINATING
workflow skeleton.  Checkpoint state is persisted via AgentStateStore at the
entry of each node so a crashed agent can resume from its last saved phase.

TASK-2.5.1 — Add checkpoint restore logic via ``resume_task()``.  When an
existing checkpoint is found, the full ``AgentWorkflowState`` is deserialised
from ``AgentCheckpoint.workflow_state`` (JSON) and the workflow graph is
entered at the node corresponding to the saved phase, skipping all earlier
nodes.

TASK-2.5.2 — Add token budget tracking and Sonnet → Opus escalation.
``build_workflow()`` now accepts ``governance``, ``default_model``,
``escalation_model``, and ``llm_call`` parameters.  A budget-checking wrapper
is applied to every node: at the start of each node the wrapper compares
``state["token_count"]`` against ``GovernancePolicy.token_budget_per_task``
and either escalates the model or raises ``FatalError``.  ``llm_call`` is an
injectable callback ``(model: str) -> int`` that returns simulated/real token
usage after each node's LLM interaction, making the behaviour testable without
live API calls.

Open Question 1 Resolution
--------------------------
``langgraph==1.1.2`` ships only ``MemorySaver`` in its core package
(``langgraph.checkpoint.memory``).  ``SqliteSaver`` and ``PostgresSaver``
require the separate ``langgraph-checkpoint-sqlite`` /
``langgraph-checkpoint-postgres`` packages which are not in the project's
dependencies.  ``MemorySaver`` is used here as the LangGraph-native
checkpointer; AgentStateStore-level persistence (the acceptance-criteria
requirement) is implemented directly in each node.  A persistent saver can be
injected via the ``checkpointer`` parameter of ``build_workflow()`` in
Milestone 2.5 without changing node logic.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Coroutine
from typing import Any

import structlog
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from tractable.agent.nodes.coordinate import make_coordinating_node
from tractable.agent.nodes.execute import make_executing_node
from tractable.agent.nodes.plan import make_planning_node
from tractable.agent.nodes.review import (
    DONE_EDGE,
    RETRY_EDGE,
    make_reviewing_node,
)
from tractable.agent.state import AgentWorkflowState
from tractable.errors import FatalError
from tractable.protocols.agent_state_store import AgentStateStore
from tractable.protocols.code_graph import CodeGraph
from tractable.protocols.tool import Tool
from tractable.types.config import GovernancePolicy
from tractable.types.enums import TaskPhase

_log = structlog.get_logger()

# Default model names — used in build_workflow() and resume_task() defaults.
_DEFAULT_MODEL = "claude-sonnet-4-6"
_ESCALATION_MODEL = "claude-opus-4-6"

# ── Node name constants ────────────────────────────────────────────────────

_NODE_PLANNING = "PLANNING"
_NODE_EXECUTING = "EXECUTING"
_NODE_REVIEWING = "REVIEWING"
_NODE_COORDINATING = "COORDINATING"

# Phase → entry node when restoring from a checkpoint.
# A checkpoint with phase=PLANNING means PLANNING is done (plan is in
# workflow_state); the workflow resumes at EXECUTING.
# A checkpoint with phase=REVIEWING means REVIEWING has not yet completed;
# the workflow resumes AT REVIEWING (skipping PLANNING and EXECUTING).
_RESTORE_ENTRY: dict[str, str] = {
    str(TaskPhase.PLANNING): _NODE_EXECUTING,
    str(TaskPhase.EXECUTING): _NODE_REVIEWING,
    str(TaskPhase.REVIEWING): _NODE_REVIEWING,
    str(TaskPhase.COORDINATING): _NODE_COORDINATING,
}


def choose_entry_node(state: AgentWorkflowState) -> str:
    """Return the first node to execute based on restore state.

    Returns the saved phase's entry point if ``resume_from`` is set;
    otherwise returns ``PLANNING`` for a fresh start.
    """
    resume = state.get("resume_from")
    if resume is not None:
        return _RESTORE_ENTRY.get(str(resume), _NODE_PLANNING)
    return _NODE_PLANNING


# ── Token budget helpers ───────────────────────────────────────────────────


def check_token_budget(
    state: AgentWorkflowState,
    governance: GovernancePolicy,
    escalation_model: str,
) -> dict[str, Any]:
    """Check token budget at node entry; return model-update dict or raise.

    Returns
    -------
    dict
        ``{"current_model": escalation_model}`` when the model is escalated
        from Sonnet to Opus.  Empty dict when the budget has not been exceeded.

    Raises
    ------
    FatalError
        When the budget is exceeded and the current model is already Opus
        (or any non-Sonnet model).
    """
    token_count: int = state["token_count"]
    budget: int = governance.token_budget_per_task
    current_model: str = state["current_model"]

    if token_count <= budget:
        return {}

    agent_id: str = state["agent_id"]
    task_id: str = state["task_id"]

    _log.warning(
        "token_budget_exceeded",
        agent_id=agent_id,
        task_id=task_id,
        tokens_used=token_count,
        budget=budget,
    )

    if "sonnet" in current_model.lower():
        _log.info(
            "model_escalated",
            agent_id=agent_id,
            task_id=task_id,
            **{"from": current_model, "to": escalation_model},
        )
        return {"current_model": escalation_model}

    raise FatalError(
        "Token budget exhausted on Opus model; task failed. Checkpoint preserved."
    )


def _wrap_with_budget_check(
    node_fn: Callable[[AgentWorkflowState], Coroutine[Any, Any, dict[str, Any]]],
    governance: GovernancePolicy,
    escalation_model: str,
    llm_call: Callable[[str], int] | None,
) -> Callable[[AgentWorkflowState], Coroutine[Any, Any, dict[str, Any]]]:
    """Wrap a node with token-budget checking and optional LLM token tracking.

    The wrapper:
    1. Calls ``check_token_budget`` at the start of the node — potentially
       escalating the model or raising ``FatalError``.
    2. Runs the underlying node with the (possibly updated) state.
    3. If ``llm_call`` is provided, invokes it with the active model name to
       obtain simulated/real token usage and updates ``token_count`` in the
       returned state dict.

    Parameters
    ----------
    node_fn:
        The original async node function to wrap.
    governance:
        Governance policy providing ``token_budget_per_task``.
    escalation_model:
        Model name to switch to when the Sonnet budget is exceeded.
    llm_call:
        Optional callback ``(model_name: str) -> tokens_used: int`` invoked
        once per node after the node's logic runs.  Used in tests to simulate
        LLM token usage without live API calls.
    """

    async def wrapper(state: AgentWorkflowState) -> dict[str, Any]:
        # Step 1: Budget check — may escalate model or raise FatalError.
        budget_updates: dict[str, Any] = check_token_budget(
            state, governance, escalation_model
        )

        # Build effective state with updated model if escalation happened.
        effective_state: AgentWorkflowState = (
            {**state, **budget_updates}  # type: ignore[misc]
            if budget_updates
            else state
        )

        # Step 2: Run the node.
        result: dict[str, Any] = await node_fn(effective_state)

        # Step 3: Simulate/record LLM token usage for this node.
        if llm_call is not None:
            active_model: str = effective_state["current_model"]
            tokens_used: int = llm_call(active_model)
            result = {
                **result,
                "token_count": effective_state["token_count"] + tokens_used,
            }

        # Merge model escalation into the result so LangGraph persists it.
        if budget_updates:
            result = {**budget_updates, **result}

        return result

    return wrapper


# ── Public API ─────────────────────────────────────────────────────────────


def build_workflow(
    tools: dict[str, Tool],
    state_store: AgentStateStore,
    graph: CodeGraph,
    *,
    governance: GovernancePolicy | None = None,
    default_model: str = _DEFAULT_MODEL,
    escalation_model: str = _ESCALATION_MODEL,
    llm_call: Callable[[str], int] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph:
    """Construct and compile the four-node LangGraph agent workflow.

    Parameters
    ----------
    tools:
        Tool name → Tool mapping.  Injected into node closures so real tools
        (Milestone 2.4) can be substituted without modifying node logic.
    state_store:
        AgentStateStore implementation used to save checkpoints at each node.
    graph:
        CodeGraph queried by the PLANNING node for repository context.
    governance:
        Governance policy.  When supplied, each node is wrapped with a
        token-budget check that may escalate the model or raise ``FatalError``.
        When ``None``, no budget enforcement is applied.
    default_model:
        LLM model name used for fresh workflow runs.  Stored in
        ``AgentWorkflowState.current_model``.
    escalation_model:
        LLM model name to escalate to when the Sonnet token budget is exceeded.
    llm_call:
        Optional callback ``(model_name: str) -> tokens_used: int``.  Called
        once per node after the node's logic to simulate or record LLM token
        usage.  Primarily for testing without live API calls.
    checkpointer:
        Optional LangGraph checkpoint saver.  Defaults to ``MemorySaver``.
        Provide a ``SqliteSaver`` or ``PostgresSaver`` (Milestone 2.5) for
        durable persistence across process restarts.

    Returns
    -------
    CompiledStateGraph
        Ready-to-invoke LangGraph workflow.
    """
    builder: StateGraph = StateGraph(AgentWorkflowState)

    planning_node = make_planning_node(tools, state_store, graph)
    executing_node = make_executing_node(tools, state_store)
    reviewing_node = make_reviewing_node(tools, state_store)
    coordinating_node = make_coordinating_node(tools, state_store)

    # Wrap nodes with budget checking when governance is provided.
    if governance is not None:
        planning_node = _wrap_with_budget_check(
            planning_node, governance, escalation_model, llm_call
        )
        executing_node = _wrap_with_budget_check(
            executing_node, governance, escalation_model, llm_call
        )
        reviewing_node = _wrap_with_budget_check(
            reviewing_node, governance, escalation_model, llm_call
        )
        coordinating_node = _wrap_with_budget_check(
            coordinating_node, governance, escalation_model, llm_call
        )

    builder.add_node(_NODE_PLANNING, planning_node)
    builder.add_node(_NODE_EXECUTING, executing_node)
    builder.add_node(_NODE_REVIEWING, reviewing_node)
    builder.add_node(_NODE_COORDINATING, coordinating_node)

    # Entry point: conditional from START based on resume_from field.
    # Fresh runs → PLANNING; restored runs → the appropriate phase node.
    builder.add_conditional_edges(
        START,
        lambda state: choose_entry_node(state),
        {
            _NODE_PLANNING: _NODE_PLANNING,
            _NODE_EXECUTING: _NODE_EXECUTING,
            _NODE_REVIEWING: _NODE_REVIEWING,
            _NODE_COORDINATING: _NODE_COORDINATING,
        },
    )

    builder.add_edge(_NODE_PLANNING, _NODE_EXECUTING)
    builder.add_edge(_NODE_EXECUTING, _NODE_REVIEWING)

    # Conditional edge from REVIEWING: retry EXECUTING on gate failure,
    # proceed to COORDINATING on success.
    # Use a lambda (no annotations) so LangGraph's get_type_hints() call at
    # graph-construction time does not try to evaluate the 'AgentWorkflowState'
    # forward reference, which is only resolvable in this module's scope.
    builder.add_conditional_edges(
        _NODE_REVIEWING,
        lambda state: DONE_EDGE if state["error"] is None else RETRY_EDGE,
        {RETRY_EDGE: _NODE_EXECUTING, DONE_EDGE: _NODE_COORDINATING},
    )

    builder.add_edge(_NODE_COORDINATING, END)

    effective_checkpointer: BaseCheckpointSaver[Any] = (
        checkpointer if checkpointer is not None else MemorySaver()
    )
    return builder.compile(checkpointer=effective_checkpointer)


async def resume_task(
    agent_id: str,
    task_id: str,
    task_description: str,
    state_store: AgentStateStore,
    tools: dict[str, Tool],
    graph: CodeGraph,
    *,
    governance: GovernancePolicy | None = None,
    default_model: str = _DEFAULT_MODEL,
    escalation_model: str = _ESCALATION_MODEL,
    llm_call: Callable[[str], int] | None = None,
    config: dict[str, Any] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> dict[str, Any]:
    """Start or resume a task, restoring from checkpoint if one exists.

    Checks ``AgentStateStore.get_checkpoint(agent_id, task_id)``:

    - **Checkpoint found**: deserialises ``checkpoint.workflow_state`` (JSON)
      into an ``AgentWorkflowState``, sets ``resume_from`` to the checkpoint's
      phase, logs ``event="checkpoint_restored"``, and invokes the workflow
      starting at the node corresponding to that phase.
    - **No checkpoint**: builds a fresh initial state and starts from
      PLANNING.

    Parameters
    ----------
    agent_id:
        Identity of the agent resuming the task.
    task_id:
        Identifier for the task being resumed or started.
    task_description:
        Human-readable task description used for fresh starts.  Ignored when
        restoring from a checkpoint (the description is already in
        ``workflow_state``).
    state_store:
        AgentStateStore used to look up the checkpoint.
    tools:
        Tool mapping injected into the workflow.
    graph:
        CodeGraph injected into the PLANNING node.
    governance:
        Governance policy forwarded to ``build_workflow()``.
    default_model:
        Default LLM model for fresh starts; also used when restoring a
        checkpoint that predates the ``current_model`` field.
    escalation_model:
        Escalation model forwarded to ``build_workflow()``.
    llm_call:
        Optional LLM token-usage callback forwarded to ``build_workflow()``.
    config:
        LangGraph invocation config (e.g. ``{"configurable": {"thread_id": …}}``).
        Defaults to ``{"configurable": {"thread_id": task_id}}``.
    checkpointer:
        Optional LangGraph checkpoint saver; forwarded to ``build_workflow()``.

    Returns
    -------
    dict[str, Any]
        Final LangGraph workflow state after completion.
    """
    if config is None:
        config = {"configurable": {"thread_id": task_id}}

    checkpoint = await state_store.get_checkpoint(agent_id, task_id)

    if checkpoint is not None:
        # Restore from checkpoint: deserialise stored workflow state.
        try:
            stored: dict[str, Any] = json.loads(checkpoint.workflow_state)
        except (json.JSONDecodeError, ValueError):
            stored = {}

        # Build the restored AgentWorkflowState, injecting resume_from so the
        # graph's entry router skips already-completed phases.
        initial_state = AgentWorkflowState(
            agent_id=str(stored.get("agent_id", agent_id)),
            task_id=str(stored.get("task_id", task_id)),
            task_description=str(stored.get("task_description", task_description)),
            phase=TaskPhase(stored.get("phase", TaskPhase.PLANNING)),
            plan=list(stored.get("plan", [])),
            files_changed=list(stored.get("files_changed", [])),
            test_results=dict(stored.get("test_results", {})),
            pr_url=stored.get("pr_url"),
            error=stored.get("error"),
            token_count=int(stored.get("token_count", 0)),
            current_model=str(stored.get("current_model", default_model)),
            messages=list(stored.get("messages", [])),
            resume_from=str(checkpoint.phase),
        )

        _log.info(
            "checkpoint_restored",
            level="info",
            agent_id=agent_id,
            task_id=task_id,
            phase=str(checkpoint.phase),
        )
    else:
        # No checkpoint: start from PLANNING.
        initial_state = AgentWorkflowState(
            agent_id=agent_id,
            task_id=task_id,
            task_description=task_description,
            phase=TaskPhase.PLANNING,
            plan=[],
            files_changed=[],
            test_results={},
            pr_url=None,
            error=None,
            token_count=0,
            current_model=default_model,
            messages=[],
            resume_from=None,
        )

    wf = build_workflow(
        tools=tools,
        state_store=state_store,
        graph=graph,
        governance=governance,
        default_model=default_model,
        escalation_model=escalation_model,
        llm_call=llm_call,
        checkpointer=checkpointer,
    )
    result: dict[str, Any] = await wf.ainvoke(initial_state, config=config)
    return result
