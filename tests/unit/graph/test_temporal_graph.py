"""Unit tests for FalkorDBTemporalGraph.

All tests use a mocked FalkorDBClient — no live FalkorDB required.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
import structlog.testing

from tractable.errors import RecoverableError
from tractable.graph.client import FalkorDBClient
from tractable.graph.temporal_graph import FalkorDBTemporalGraph, _inject_current_filter
from tractable.types.enums import ChangeRisk, ChangeSource
from tractable.types.temporal import TemporalMutation

# ── Fixture ────────────────────────────────────────────────────────────────────


def make_graph(
    execute_return: list[dict[str, Any]] | None = None,
) -> tuple[FalkorDBTemporalGraph, AsyncMock]:
    """Return a (graph, mock_client) pair for unit testing."""
    mock = AsyncMock()
    mock.execute = AsyncMock(return_value=execute_return or [])
    mock.execute_write = AsyncMock(return_value=[])
    graph = FalkorDBTemporalGraph(cast(FalkorDBClient, mock))
    return graph, mock


def entity_row(
    entity_id: str = "e-1",
    version_id: str = "v-1",
    valid_until: str | None = None,
) -> dict[str, Any]:
    """Build a minimal entity property row as FalkorDB would return it."""
    return {
        "id": entity_id,
        "version_id": version_id,
        "kind": "function",
        "name": "get_user",
        "qualified_name": "myrepo.get_user",
        "repo": "myrepo",
        "file_path": "src/users.py",
        "valid_from": "2026-03-19T10:00:00+00:00",
        "valid_until": valid_until,
        "observed_at": "2026-03-19T10:00:00+00:00",
        "change_source": "initial_ingestion",
        "commit_sha": None,
        "agent_id": None,
        "superseded_by": None,
    }


# ── _inject_current_filter ─────────────────────────────────────────────────────


class TestInjectCurrentFilter:
    def test_adds_where_before_return(self) -> None:
        q = "MATCH (e:Entity) RETURN e"
        result = _inject_current_filter(q)
        assert "WHERE e.valid_until IS NULL" in result
        assert result.index("WHERE") < result.index("RETURN")

    def test_prepends_to_existing_where(self) -> None:
        q = "MATCH (e:Entity) WHERE e.repo = 'x' RETURN e"
        result = _inject_current_filter(q)
        assert "e.valid_until IS NULL AND e.repo = 'x'" in result

    def test_no_return_appends_at_end(self) -> None:
        q = "MATCH (e:Entity)"
        result = _inject_current_filter(q)
        assert result.endswith("WHERE e.valid_until IS NULL")


# ── get_current_entity ────────────────────────────────────────────────────────


class TestGetCurrentEntity:
    @pytest.mark.asyncio
    async def test_returns_entity_when_found(self) -> None:
        graph, _ = make_graph(execute_return=[entity_row()])
        result = await graph.get_current_entity("e-1")
        assert result is not None
        assert result.id == "e-1"
        assert result.version_id == "v-1"
        assert result.is_current is True

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self) -> None:
        graph, _ = make_graph(execute_return=[])
        result = await graph.get_current_entity("missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_passes_entity_id_as_param(self) -> None:
        graph, mock = make_graph(execute_return=[entity_row()])
        await graph.get_current_entity("e-42")
        call_args = mock.execute.call_args
        assert call_args[0][1] == {"id": "e-42"}

    @pytest.mark.asyncio
    async def test_entity_with_valid_until_is_not_current(self) -> None:
        row = entity_row(valid_until="2026-03-20T00:00:00+00:00")
        graph, _ = make_graph(execute_return=[row])
        result = await graph.get_current_entity("e-1")
        assert result is not None
        assert result.is_current is False


# ── query_current ─────────────────────────────────────────────────────────────


class TestQueryCurrent:
    @pytest.mark.asyncio
    async def test_injects_valid_until_filter(self) -> None:
        graph, mock = make_graph()
        await graph.query_current("MATCH (e:Entity) RETURN e")
        sent_cypher: str = mock.execute.call_args[0][0]
        assert "e.valid_until IS NULL" in sent_cypher

    @pytest.mark.asyncio
    async def test_passes_params_through(self) -> None:
        graph, mock = make_graph()
        await graph.query_current("MATCH (e:Entity) RETURN e", {"repo": "svc"})
        assert mock.execute.call_args[0][1] == {"repo": "svc"}


# ── apply_mutations — create ───────────────────────────────────────────────────


class TestApplyMutationsCreate:
    @pytest.mark.asyncio
    async def test_create_entity_calls_execute_write(self) -> None:
        graph, mock = make_graph()
        mutation = TemporalMutation(
            operation="create_entity",
            payload={
                "id": "e-new",
                "version_id": "v-1",
                "kind": "function",
                "name": "foo",
                "qualified_name": "repo.foo",
                "repo": "myrepo",
                "file_path": "src/foo.py",
            },
        )
        result = await graph.apply_mutations(
            [mutation], ChangeSource.INITIAL_INGESTION
        )
        mock.execute_write.assert_called_once()
        cypher: str = mock.execute_write.call_args[0][0]
        assert "CREATE (e:Entity" in cypher
        assert result.entities_created == 1
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_create_passes_entity_id_in_params(self) -> None:
        graph, mock = make_graph()
        mutation = TemporalMutation(
            operation="create_entity",
            payload={"id": "e-99", "version_id": "v-1"},
        )
        await graph.apply_mutations([mutation], ChangeSource.HUMAN_COMMIT)
        params: dict[str, Any] = mock.execute_write.call_args[0][1]
        assert params["id"] == "e-99"

    @pytest.mark.asyncio
    async def test_result_timestamp_is_utc(self) -> None:
        graph, _ = make_graph()
        mutation = TemporalMutation(operation="create_entity", payload={"id": "e-1"})
        result = await graph.apply_mutations([mutation], ChangeSource.AGENT_COMMIT)
        assert result.timestamp.tzinfo is not None


# ── apply_mutations — update ───────────────────────────────────────────────────


class TestApplyMutationsUpdate:
    @pytest.mark.asyncio
    async def test_update_calls_execute_write_twice(self) -> None:
        """update_entity must close old version and create new one."""
        graph, mock = make_graph()
        mutation = TemporalMutation(
            operation="update_entity",
            entity_id="e-1",
            payload={"version_id": "v-2", "id": "e-1", "kind": "function",
                     "name": "foo", "qualified_name": "repo.foo",
                     "repo": "myrepo", "file_path": "src/foo.py"},
        )
        result = await graph.apply_mutations([mutation], ChangeSource.HUMAN_COMMIT)
        assert mock.execute_write.call_count == 2
        assert result.entities_updated == 1
        assert result.entities_created == 0

    @pytest.mark.asyncio
    async def test_update_first_call_sets_valid_until(self) -> None:
        graph, mock = make_graph()
        mutation = TemporalMutation(
            operation="update_entity",
            entity_id="e-1",
            payload={"version_id": "v-2"},
        )
        await graph.apply_mutations([mutation], ChangeSource.HUMAN_COMMIT)
        first_call_cypher: str = mock.execute_write.call_args_list[0][0][0]
        assert "SET e.valid_until" in first_call_cypher
        assert "WHERE e.valid_until IS NULL" in first_call_cypher

    @pytest.mark.asyncio
    async def test_update_second_call_creates_entity(self) -> None:
        graph, mock = make_graph()
        mutation = TemporalMutation(
            operation="update_entity",
            entity_id="e-1",
            payload={"version_id": "v-2"},
        )
        await graph.apply_mutations([mutation], ChangeSource.HUMAN_COMMIT)
        second_call_cypher: str = mock.execute_write.call_args_list[1][0][0]
        assert "CREATE (e:Entity" in second_call_cypher


# ── apply_mutations — delete ───────────────────────────────────────────────────


class TestApplyMutationsDelete:
    @pytest.mark.asyncio
    async def test_delete_closes_current_version(self) -> None:
        graph, mock = make_graph()
        mutation = TemporalMutation(
            operation="delete_entity",
            entity_id="e-1",
        )
        result = await graph.apply_mutations([mutation], ChangeSource.HUMAN_COMMIT)
        mock.execute_write.assert_called_once()
        cypher: str = mock.execute_write.call_args[0][0]
        assert "SET e.valid_until" in cypher
        assert "WHERE e.valid_until IS NULL" in cypher
        assert result.entities_deleted == 1

    @pytest.mark.asyncio
    async def test_delete_passes_entity_id(self) -> None:
        graph, mock = make_graph()
        mutation = TemporalMutation(operation="delete_entity", entity_id="e-42")
        await graph.apply_mutations([mutation], ChangeSource.HUMAN_COMMIT)
        params: dict[str, Any] = mock.execute_write.call_args[0][1]
        assert params["id"] == "e-42"


# ── impact_analysis_current ───────────────────────────────────────────────────


class TestImpactAnalysisCurrent:
    @pytest.mark.asyncio
    async def test_returns_impact_report_structure(self) -> None:
        """Returns ImpactReport without error even on an empty graph."""
        graph, _ = make_graph(execute_return=[])
        report = await graph.impact_analysis_current(["e-1"], depth=2)
        assert report.directly_affected == []
        assert report.transitively_affected == []
        assert report.risk_level == ChangeRisk.LOW

    @pytest.mark.asyncio
    async def test_direct_neighbors_in_directly_affected(self) -> None:
        neighbor_row: dict[str, Any] = {
            "id": "e-2",
            "kind": "function",
            "name": "bar",
            "repo": "myrepo",
            "file_path": "src/bar.py",
            "confidence": 0.9,
        }
        graph, mock = make_graph(execute_return=[neighbor_row])
        report = await graph.impact_analysis_current(["e-1"], depth=1)
        assert len(report.directly_affected) == 1
        assert report.directly_affected[0].id == "e-2"

    @pytest.mark.asyncio
    async def test_low_confidence_edges_filtered_out(self) -> None:
        low_conf_row: dict[str, Any] = {
            "id": "e-low",
            "kind": "function",
            "name": "low",
            "repo": "myrepo",
            "file_path": "src/low.py",
            "confidence": 0.1,
        }
        graph, _ = make_graph(execute_return=[low_conf_row])
        report = await graph.impact_analysis_current(["e-1"], depth=1, min_confidence=0.5)
        assert report.directly_affected == []

    @pytest.mark.asyncio
    async def test_risk_escalates_with_affected_count(self) -> None:
        rows: list[dict[str, Any]] = [
            {"id": f"e-{i}", "kind": "fn", "name": f"f{i}",
             "repo": "r", "file_path": f"f{i}.py", "confidence": 1.0}
            for i in range(25)
        ]
        graph, _ = make_graph(execute_return=rows)
        report = await graph.impact_analysis_current(["start"], depth=1)
        assert report.risk_level == ChangeRisk.CRITICAL


# ── error handling ────────────────────────────────────────────────────────────


class TestMutationErrors:
    @pytest.mark.asyncio
    async def test_error_is_captured_not_raised(self) -> None:
        graph, mock = make_graph()
        mock.execute_write = AsyncMock(side_effect=RuntimeError("db error"))
        mutation = TemporalMutation(operation="create_entity", payload={"id": "e-1"})
        result = await graph.apply_mutations([mutation], ChangeSource.HUMAN_COMMIT)
        assert len(result.errors) == 1
        assert "db error" in result.errors[0]
        assert result.entities_created == 0

    @pytest.mark.asyncio
    async def test_successful_mutations_counted_despite_earlier_error(self) -> None:
        graph, mock = make_graph()
        mock.execute_write = AsyncMock(
            side_effect=[RuntimeError("fail"), None]
        )
        mutations = [
            TemporalMutation(operation="create_entity", payload={"id": "e-1"}),
            TemporalMutation(operation="create_entity", payload={"id": "e-2"}),
        ]
        result = await graph.apply_mutations(mutations, ChangeSource.HUMAN_COMMIT)
        assert len(result.errors) == 1
        assert result.entities_created == 1


# ── AC-3: RecoverableError when _inject_current_filter returns malformed query ─


class TestInjectFilterMalformedQuery:
    @pytest.mark.asyncio
    async def test_inject_filter_malformed_query(self) -> None:
        """AC-3: query_current raises RecoverableError if the injected filter is missing."""
        mock = AsyncMock()
        mock.execute = AsyncMock(return_value=[])
        mock.execute_write = AsyncMock(return_value=[])
        graph = FalkorDBTemporalGraph(cast(FalkorDBClient, mock))

        # Patch _inject_current_filter at the module level so query_current
        # sees a result that lacks the validity filter.
        with patch(
            "tractable.graph.temporal_graph._inject_current_filter",
            return_value="MATCH (e:Entity) RETURN e",  # missing valid_until IS NULL
        ):
            with pytest.raises(RecoverableError, match="malformed query"):
                await graph.query_current("MATCH (e:Entity) RETURN e")


# ── AC-4: structlog entry on apply_mutations ───────────────────────────────────


class TestApplyMutationsLogging:
    @pytest.mark.asyncio
    async def test_mutations_applied_log_entry(self) -> None:
        """AC-4: apply_mutations emits event='mutations_applied' with mutation_count."""
        mock = AsyncMock()
        mock.execute = AsyncMock(return_value=[])  # orphan check + any reads
        mock.execute_write = AsyncMock(return_value=[])
        graph = FalkorDBTemporalGraph(cast(FalkorDBClient, mock))

        mutations = [
            TemporalMutation(operation="create_entity", payload={"id": "e-1"}),
            TemporalMutation(operation="create_entity", payload={"id": "e-2"}),
        ]
        with structlog.testing.capture_logs() as cap:
            await graph.apply_mutations(mutations, ChangeSource.HUMAN_COMMIT)

        applied = [r for r in cap if r.get("event") == "mutations_applied"]
        assert applied, f"Expected 'mutations_applied' log entry, got: {cap}"
        entry = applied[0]
        assert entry["mutation_count"] == 2
        assert entry["log_level"] == "info"

    @pytest.mark.asyncio
    async def test_mutations_applying_log_entry(self) -> None:
        """apply_mutations emits event='mutations_applying' before execution."""
        mock = AsyncMock()
        mock.execute = AsyncMock(return_value=[])
        mock.execute_write = AsyncMock(return_value=[])
        graph = FalkorDBTemporalGraph(cast(FalkorDBClient, mock))

        with structlog.testing.capture_logs() as cap:
            await graph.apply_mutations([], ChangeSource.HUMAN_COMMIT)

        applying = [r for r in cap if r.get("event") == "mutations_applying"]
        assert applying


# ── AC-6: orphan detection logging ────────────────────────────────────────────


class TestOrphanDetection:
    @pytest.mark.asyncio
    async def test_graph_orphan_detected_log_entry(self) -> None:
        """AC-6: _check_for_orphaned_entities logs event='graph_orphan_detected'."""
        orphan_rows = [{"entity_id": "orphan-123"}]
        mock = AsyncMock()
        # First execute call (orphan check) returns the orphan; subsequent calls return []
        mock.execute = AsyncMock(side_effect=[orphan_rows, [], []])
        mock.execute_write = AsyncMock(return_value=[])
        graph = FalkorDBTemporalGraph(cast(FalkorDBClient, mock))

        with structlog.testing.capture_logs() as cap:
            await graph.apply_mutations([], ChangeSource.HUMAN_COMMIT)

        orphan_entries = [r for r in cap if r.get("event") == "graph_orphan_detected"]
        assert orphan_entries, f"Expected 'graph_orphan_detected' log entry, got: {cap}"
        entry = orphan_entries[0]
        assert "orphan-123" in entry["entity_ids"]
        assert entry["log_level"] == "error"

    @pytest.mark.asyncio
    async def test_no_orphan_log_when_graph_clean(self) -> None:
        """No orphan log entry when graph has no orphaned entities."""
        mock = AsyncMock()
        mock.execute = AsyncMock(return_value=[])  # no orphans
        mock.execute_write = AsyncMock(return_value=[])
        graph = FalkorDBTemporalGraph(cast(FalkorDBClient, mock))

        with structlog.testing.capture_logs() as cap:
            await graph.apply_mutations([], ChangeSource.HUMAN_COMMIT)

        orphan_entries = [r for r in cap if r.get("event") == "graph_orphan_detected"]
        assert not orphan_entries
