# pyright: reportPrivateUsage=false
"""Gap-fill unit tests for FalkorDBTemporalGraph (TASK-3.3.4).

Covers paths not exercised by test_temporal_graph.py:
- _inject_at_filter: fallback path (no WHERE / no RETURN in query)
- apply_mutations: edge operations (create_edge, update_edge, delete_edge)
- get_changes_since: empty result returns empty ChangeSet
- get_changes_by_commit: added, removed, and empty scenarios
- impact_analysis_current: HIGH and MEDIUM risk levels
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from tractable.graph.client import FalkorDBClient
from tractable.graph.temporal_graph import FalkorDBTemporalGraph, _inject_at_filter
from tractable.types.enums import ChangeRisk, ChangeSource
from tractable.types.temporal import TemporalMutation


def _make_graph(
    execute_return: list[dict[str, Any]] | None = None,
) -> tuple[FalkorDBTemporalGraph, AsyncMock]:
    mock = AsyncMock()
    mock.execute = AsyncMock(return_value=execute_return or [])
    mock.execute_write = AsyncMock(return_value=[])
    return FalkorDBTemporalGraph(cast(FalkorDBClient, mock)), mock


def _entity_row(eid: str = "e-1", valid_until: str | None = None) -> dict[str, Any]:
    return {
        "id": eid,
        "version_id": "v-1",
        "kind": "function",
        "name": "foo",
        "qualified_name": f"repo.{eid}",
        "repo": "myrepo",
        "file_path": "src/foo.py",
        "valid_from": "2026-03-19T10:00:00+00:00",
        "valid_until": valid_until,
        "observed_at": "2026-03-19T10:00:00+00:00",
        "change_source": "human_commit",
        "commit_sha": "abc123",
        "agent_id": None,
        "superseded_by": None,
    }


# ── _inject_at_filter ──────────────────────────────────────────────────────────


class TestInjectAtFilter:
    def test_replaces_existing_current_filter(self) -> None:
        q = "MATCH (e:Entity) WHERE e.valid_until IS NULL RETURN e"
        result = _inject_at_filter(q, "2026-01-01T00:00:00+00:00")
        assert "e.valid_from <= $__at" in result

    def test_inserts_before_return_when_no_where(self) -> None:
        q = "MATCH (e:Entity) RETURN e"
        result = _inject_at_filter(q, "2026-01-01T00:00:00+00:00")
        assert "e.valid_from <= $__at" in result
        assert result.index("WHERE") < result.index("RETURN")

    def test_appends_when_no_where_and_no_return(self) -> None:
        """Fallback: inject temporal condition at end of query."""
        q = "MATCH (e:Entity)"
        result = _inject_at_filter(q, "2026-01-01T00:00:00+00:00")
        assert "e.valid_from <= $__at" in result
        assert result.startswith("MATCH (e:Entity)")


# ── apply_mutations: edge operations ──────────────────────────────────────────


class TestApplyMutationsEdges:
    @pytest.mark.asyncio
    async def test_create_edge_increments_edges_created(self) -> None:
        graph, mock = _make_graph()
        mutation = TemporalMutation(
            operation="create_edge",
            payload={
                "source_entity_id": "e-src",
                "target_entity_id": "e-tgt",
                "relationship": "CALLS",
                "confidence": 0.9,
            },
        )
        result = await graph.apply_mutations([mutation], ChangeSource.HUMAN_COMMIT)
        mock.execute_write.assert_called_once()
        cypher: str = mock.execute_write.call_args[0][0]
        assert "CREATE" in cypher
        assert result.edges_created == 1
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_delete_edge_increments_edges_deleted(self) -> None:
        graph, mock = _make_graph()
        mutation = TemporalMutation(
            operation="delete_edge",
            edge_id="edge-1",
            payload={},
        )
        result = await graph.apply_mutations([mutation], ChangeSource.HUMAN_COMMIT)
        mock.execute_write.assert_called_once()
        cypher: str = mock.execute_write.call_args[0][0]
        assert "SET r.valid_until" in cypher
        assert result.edges_deleted == 1

    @pytest.mark.asyncio
    async def test_update_edge_closes_old_and_creates_new(self) -> None:
        graph, mock = _make_graph()
        mutation = TemporalMutation(
            operation="update_edge",
            edge_id="edge-1",
            payload={
                "version_id": "v-2",
                "source_entity_id": "e-src",
                "target_entity_id": "e-tgt",
            },
        )
        result = await graph.apply_mutations([mutation], ChangeSource.HUMAN_COMMIT)
        assert mock.execute_write.call_count == 2
        assert result.edges_deleted == 1
        assert result.edges_created == 1


# ── get_changes_since: empty result ───────────────────────────────────────────


class TestGetChangesSinceEmpty:
    @pytest.mark.asyncio
    async def test_empty_window_returns_empty_changeset(self) -> None:
        """get_changes_since with no changed entities returns empty ChangeSet."""
        graph, _ = _make_graph(execute_return=[])
        since = datetime(2026, 1, 1, tzinfo=UTC)
        changeset = await graph.get_changes_since(since)
        assert changeset.entities_added == []
        assert changeset.entities_modified == []
        assert changeset.entities_removed == []
        assert changeset.commits == []

    @pytest.mark.asyncio
    async def test_empty_result_time_range_is_set(self) -> None:
        """get_changes_since populates time_range_start even with no results."""
        graph, _ = _make_graph(execute_return=[])
        since = datetime(2026, 1, 1, tzinfo=UTC)
        changeset = await graph.get_changes_since(since)
        assert changeset.time_range_start == since


# ── get_changes_by_commit ──────────────────────────────────────────────────────


class TestGetChangesByCommit:
    @pytest.mark.asyncio
    async def test_returns_added_entity_for_commit(self) -> None:
        graph, _ = _make_graph(execute_return=[_entity_row("e-1")])
        changeset = await graph.get_changes_by_commit("abc123")
        assert len(changeset.entities_added) == 1
        assert changeset.entities_added[0].id == "e-1"

    @pytest.mark.asyncio
    async def test_empty_commit_returns_empty_changeset(self) -> None:
        graph, _ = _make_graph(execute_return=[])
        changeset = await graph.get_changes_by_commit("sha-not-found")
        assert changeset.entities_added == []
        assert changeset.entities_removed == []

    @pytest.mark.asyncio
    async def test_closed_entity_classified_as_removed(self) -> None:
        row = _entity_row("e-closed", valid_until="2026-03-20T00:00:00+00:00")
        graph, _ = _make_graph(execute_return=[row])
        changeset = await graph.get_changes_by_commit("abc123")
        assert len(changeset.entities_removed) == 1
        assert changeset.entities_removed[0].id == "e-closed"


# ── impact_analysis_current: risk level thresholds ────────────────────────────


class TestImpactAnalysisRiskLevels:
    @pytest.mark.asyncio
    async def test_high_risk_when_11_to_20_entities_affected(self) -> None:
        rows: list[dict[str, Any]] = [
            {
                "id": f"e-{i}",
                "kind": "fn",
                "name": f"f{i}",
                "repo": "r",
                "file_path": f"f{i}.py",
                "confidence": 1.0,
            }
            for i in range(15)
        ]
        graph, _ = _make_graph(execute_return=rows)
        report = await graph.impact_analysis_current(["start"], depth=1)
        assert report.risk_level == ChangeRisk.HIGH

    @pytest.mark.asyncio
    async def test_medium_risk_when_4_to_10_entities_affected(self) -> None:
        rows: list[dict[str, Any]] = [
            {
                "id": f"e-{i}",
                "kind": "fn",
                "name": f"f{i}",
                "repo": "r",
                "file_path": f"f{i}.py",
                "confidence": 1.0,
            }
            for i in range(5)
        ]
        graph, _ = _make_graph(execute_return=rows)
        report = await graph.impact_analysis_current(["start"], depth=1)
        assert report.risk_level == ChangeRisk.MEDIUM
