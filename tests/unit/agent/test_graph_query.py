"""Unit tests for tractable/agent/tools/graph_query.py.

TASK-2.4.2 acceptance criteria:
1. query_current with 5 rows returns ToolResult(success=True, output=JSON array of 5 dicts).
2. query_current with 150 rows returns 100 rows + "truncated": true.
3. depth > 3 in get_neighborhood is capped to 3; mock called with depth=3.
4. TransientError from query_current is re-raised (not wrapped in ToolResult).
5. get_neighborhood with single-node subgraph returns incomplete=True + reason="empty_neighborhood".
6. impact_analysis always returns heuristic_edges_present=True + calls_edges_traversed=False.
7. pyright strict-mode clean (verified separately).
8. ruff clean (verified separately).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tractable.agent.tools.graph_query import GraphQueryTool
from tractable.errors import TransientError
from tractable.types.enums import ChangeRisk
from tractable.types.graph import GraphEntity, ImpactReport, Subgraph

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_entity(entity_id: str = "e1") -> GraphEntity:
    return GraphEntity(id=entity_id, kind="function", name="foo", repo="r", file_path="f.py")


def _make_tool(graph: MagicMock) -> GraphQueryTool:
    return GraphQueryTool(
        graph=graph,
        agent_id="agent-test",
        task_id="task-test",
        repo="test/repo",
    )


def _mock_graph() -> MagicMock:
    g = MagicMock()
    g.query_current = AsyncMock()
    g.get_neighborhood = AsyncMock()
    g.impact_analysis_current = AsyncMock()
    return g


# ── AC-1: query_current with 5 rows ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_current_returns_5_rows() -> None:
    rows = [{"n": i} for i in range(5)]
    graph = _mock_graph()
    graph.query_current.return_value = rows

    result = await _make_tool(graph).invoke(
        {"operation": "query_current", "cypher": "MATCH (n) RETURN n", "params": {}}
    )

    assert result.success is True
    output = json.loads(result.output)
    assert output["rows"] == rows
    assert "truncated" not in output


# ── AC-2: query_current with 150 rows returns 100 + truncated=true ───────────


@pytest.mark.asyncio
async def test_query_current_caps_at_100_rows() -> None:
    rows = [{"n": i} for i in range(150)]
    graph = _mock_graph()
    graph.query_current.return_value = rows

    result = await _make_tool(graph).invoke(
        {"operation": "query_current", "cypher": "MATCH (n) RETURN n", "params": {}}
    )

    assert result.success is True
    output = json.loads(result.output)
    assert len(output["rows"]) == 100
    assert output["truncated"] is True


# ── AC-3: depth > 3 is capped to 3 ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_neighborhood_caps_depth_to_3() -> None:
    entity = _make_entity("target")
    subgraph = Subgraph(nodes=[entity, _make_entity("neighbor")], edges=[])
    graph = _mock_graph()
    graph.get_neighborhood.return_value = subgraph

    await _make_tool(graph).invoke(
        {"operation": "get_neighborhood", "entity_id": "target", "depth": 99}
    )

    graph.get_neighborhood.assert_called_once_with("target", depth=3)


# ── AC-4: TransientError from query_current is re-raised ─────────────────────


@pytest.mark.asyncio
async def test_query_current_reraises_transient_error() -> None:
    graph = _mock_graph()
    graph.query_current.side_effect = TransientError("graph unavailable")

    with pytest.raises(TransientError):
        await _make_tool(graph).invoke(
            {"operation": "query_current", "cypher": "MATCH (n) RETURN n", "params": {}}
        )


# ── AC-5: single-node subgraph returns incomplete=true ───────────────────────


@pytest.mark.asyncio
async def test_get_neighborhood_empty_returns_incomplete() -> None:
    entity = _make_entity("unknown-entity")
    subgraph = Subgraph(nodes=[entity], edges=[])
    graph = _mock_graph()
    graph.get_neighborhood.return_value = subgraph

    result = await _make_tool(graph).invoke(
        {"operation": "get_neighborhood", "entity_id": "unknown-entity", "depth": 2}
    )

    assert result.success is True
    output = json.loads(result.output)
    assert output["incomplete"] is True
    assert output["reason"] == "empty_neighborhood"


# ── AC-6: impact_analysis always includes limitation flags ───────────────────


@pytest.mark.asyncio
async def test_impact_analysis_includes_limitation_flags() -> None:
    report = ImpactReport(
        directly_affected=[_make_entity()],
        transitively_affected=[],
        affected_repos=["test/repo"],
        risk_level=ChangeRisk.LOW,
    )
    graph = _mock_graph()
    graph.impact_analysis_current.return_value = report

    result = await _make_tool(graph).invoke(
        {
            "operation": "impact_analysis",
            "entity_ids": ["e1"],
            "depth": 3,
            "min_confidence": 0.5,
        }
    )

    assert result.success is True
    output = json.loads(result.output)
    assert output["heuristic_edges_present"] is True
    assert output["calls_edges_traversed"] is False


# ── Additional: non-transient exception returns ToolResult(success=False) ────


@pytest.mark.asyncio
async def test_non_transient_exception_returns_failure_result() -> None:
    graph = _mock_graph()
    graph.query_current.side_effect = RuntimeError("connection reset")

    result = await _make_tool(graph).invoke(
        {"operation": "query_current", "cypher": "MATCH (n) RETURN n", "params": {}}
    )

    assert result.success is False
    assert "connection reset" in (result.error or "")


# ── Additional: depth <= 3 is NOT capped ─────────────────────────────────────


@pytest.mark.asyncio
async def test_get_neighborhood_depth_within_cap_not_changed() -> None:
    entity = _make_entity("e")
    subgraph = Subgraph(nodes=[entity, _make_entity("n2")], edges=[])
    graph = _mock_graph()
    graph.get_neighborhood.return_value = subgraph

    await _make_tool(graph).invoke({"operation": "get_neighborhood", "entity_id": "e", "depth": 2})

    graph.get_neighborhood.assert_called_once_with("e", depth=2)
