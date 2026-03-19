"""graph_query_mcp tool — exposes TemporalCodeGraph operations to the LLM.

TASK-2.4.2: Implements the ``graph_query_mcp`` tool satisfying the ``Tool``
Protocol (tech-spec.py §2.6).  Wraps three graph operations with row/depth
caps and explicit limitations disclosure.

Sources:
- tech-spec.py §2.2 — CodeGraph Protocol: get_neighborhood
- tech-spec.py §2.6 — Tool Protocol
- realtime-temporal-spec.py §B — TemporalCodeGraph: query_current, impact_analysis_current

Known limitation (phase-1-analysis.md §3.8, §4.2):
    impact_analysis_current() traverses only DEPENDS_ON and IMPORTS edges.
    CALLS edges are heuristic in Phase 1 and are NOT traversed.  The tool
    always surfaces this as ``heuristic_edges_present=True`` and
    ``calls_edges_traversed=False`` in the output JSON.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import structlog

from tractable.errors import TransientError
from tractable.protocols.tool import ToolResult
from tractable.types.graph import ImpactReport, Subgraph

_log = structlog.get_logger()

_ROW_CAP = 100
_DEPTH_CAP = 3


# ── Internal protocol ──────────────────────────────────────────────────────────
#
# TemporalCodeGraph exposes query_current and impact_analysis_current.
# get_neighborhood lives on CodeGraph.  We define a narrow structural protocol
# that captures just the three methods this tool requires so that pyright can
# type-check call sites without needing to import the concrete class.


@runtime_checkable
class _QueryableGraph(Protocol):
    async def query_current(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> Sequence[dict[str, Any]]: ...

    async def get_neighborhood(
        self, entity_id: str, depth: int = 2, min_confidence: float = 0.7
    ) -> Subgraph: ...

    async def impact_analysis_current(
        self,
        entity_ids: Sequence[str],
        depth: int = 3,
        min_confidence: float = 0.5,
    ) -> ImpactReport: ...


# ── Tool ──────────────────────────────────────────────────────────────────────


class GraphQueryTool:
    """MCP-style tool that exposes three graph operations to the agent LLM.

    Parameters
    ----------
    graph:
        Object satisfying the ``_QueryableGraph`` protocol — in production
        this is a ``FalkorTemporalGraph`` which implements both
        ``TemporalCodeGraph`` and ``CodeGraph``.
    agent_id:
        Identifier of the running agent (included in structlog entries).
    task_id:
        Identifier of the current task (included in structlog entries).
    repo:
        Human-readable repository name (included in structlog entries).
    """

    def __init__(
        self,
        graph: _QueryableGraph,
        agent_id: str,
        task_id: str,
        repo: str,
    ) -> None:
        self._graph = graph
        self._agent_id = agent_id
        self._task_id = task_id
        self._repo = repo

    # ── Tool Protocol ────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "graph_query_mcp"

    @property
    def description(self) -> str:
        return (
            "Query the live code knowledge graph. "
            "Operations: query_current (Cypher), get_neighborhood (BFS subgraph), "
            "impact_analysis (blast-radius for proposed changes)."
        )

    async def invoke(self, params: dict[str, Any]) -> ToolResult:
        """Dispatch to the appropriate graph operation.

        Expected ``params`` keys:
        - ``operation``: one of ``"query_current"``, ``"get_neighborhood"``,
          ``"impact_analysis"``
        - Operation-specific sub-parameters (see individual handlers).
        """
        operation: str = params.get("operation", "")
        try:
            if operation == "query_current":
                return await self._query_current(params)
            if operation == "get_neighborhood":
                return await self._get_neighborhood(params)
            if operation == "impact_analysis":
                return await self._impact_analysis(params)
            return ToolResult(success=False, error=f"Unknown operation: {operation!r}")
        except TransientError:
            raise
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=str(exc))

    # ── Operations ───────────────────────────────────────────────────────────

    async def _query_current(self, params: dict[str, Any]) -> ToolResult:
        cypher: str = params.get("cypher", "")
        query_params: dict[str, Any] = params.get("params", {})

        rows = list(await self._graph.query_current(cypher, query_params))

        truncated = len(rows) > _ROW_CAP
        result_rows = rows[:_ROW_CAP]

        output: dict[str, Any] = {"rows": result_rows}
        if truncated:
            output["truncated"] = True

        return ToolResult(success=True, output=json.dumps(output))

    async def _get_neighborhood(self, params: dict[str, Any]) -> ToolResult:
        entity_id: str = params.get("entity_id", "")
        depth: int = min(int(params.get("depth", 2)), _DEPTH_CAP)

        subgraph = await self._graph.get_neighborhood(entity_id, depth=depth)

        output = subgraph.model_dump()

        # Empty neighborhood: the subgraph contains only the queried entity
        # (or zero nodes).  Signal the PLANNING node to fall back to file reads.
        other_nodes = [n for n in subgraph.nodes if n.id != entity_id]
        if not other_nodes:
            output["incomplete"] = True
            output["reason"] = "empty_neighborhood"

        return ToolResult(success=True, output=json.dumps(output))

    async def _impact_analysis(self, params: dict[str, Any]) -> ToolResult:
        entity_ids: list[str] = params.get("entity_ids", [])
        depth: int = int(params.get("depth", 3))
        min_confidence: float = float(params.get("min_confidence", 0.5))

        report = await self._graph.impact_analysis_current(
            entity_ids, depth=depth, min_confidence=min_confidence
        )

        output = report.model_dump()

        # Always surface the known Phase-1 limitation so the LLM can account
        # for the understated blast radius.
        output["heuristic_edges_present"] = True
        output["calls_edges_traversed"] = False

        _log.info(
            "impact_analysis_run",
            agent_id=self._agent_id,
            task_id=self._task_id,
            repo=self._repo,
            entity_count=len(entity_ids),
            depth=depth,
            affected_entity_count=(
                len(report.directly_affected) + len(report.transitively_affected)
            ),
        )

        return ToolResult(success=True, output=json.dumps(output))
