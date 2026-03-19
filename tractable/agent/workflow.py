# pyright: reportUnknownMemberType=false, reportArgumentType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false
"""LangGraph four-node agent workflow with checkpoint persistence.

TASK-2.3.1 — Implement the PLANNING → EXECUTING → REVIEWING → COORDINATING
workflow skeleton.  Checkpoint state is persisted via AgentStateStore at the
entry of each node so a crashed agent can resume from its last saved phase.

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

from typing import Any

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
from tractable.protocols.agent_state_store import AgentStateStore
from tractable.protocols.code_graph import CodeGraph
from tractable.protocols.tool import Tool

# ── Node name constants ────────────────────────────────────────────────────

_NODE_PLANNING = "PLANNING"
_NODE_EXECUTING = "EXECUTING"
_NODE_REVIEWING = "REVIEWING"
_NODE_COORDINATING = "COORDINATING"

# ── Public API ─────────────────────────────────────────────────────────────


def build_workflow(
    tools: dict[str, Tool],
    state_store: AgentStateStore,
    graph: CodeGraph,
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

    builder.add_node(_NODE_PLANNING, make_planning_node(tools, state_store, graph))
    builder.add_node(_NODE_EXECUTING, make_executing_node(tools, state_store))
    builder.add_node(_NODE_REVIEWING, make_reviewing_node(tools, state_store))
    builder.add_node(_NODE_COORDINATING, make_coordinating_node(tools, state_store))

    # Primary linear path.
    builder.add_edge(START, _NODE_PLANNING)
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
