"""Unit tests for FalkorDBTemporalGraph temporal query methods (TASK-1.4.3).

All tests use a mocked FalkorDBClient — no live FalkorDB required.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from tractable.graph.client import FalkorDBClient
from tractable.graph.temporal_graph import (
    FalkorDBTemporalGraph,
    _compute_changed_fields,
    _inject_at_filter,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 1, 2, 0, 0, 0, tzinfo=UTC)
T2 = datetime(2026, 1, 3, 0, 0, 0, tzinfo=UTC)


def make_graph(
    side_effects: list[list[dict[str, Any]]] | None = None,
    default_return: list[dict[str, Any]] | None = None,
) -> tuple[FalkorDBTemporalGraph, AsyncMock]:
    mock = AsyncMock()
    if side_effects is not None:
        mock.execute = AsyncMock(side_effect=side_effects)
    else:
        mock.execute = AsyncMock(return_value=default_return or [])
    mock.execute_write = AsyncMock(return_value=[])
    graph = FalkorDBTemporalGraph(cast(FalkorDBClient, mock))
    return graph, mock


def entity_row(
    entity_id: str = "e-1",
    version_id: str = "v-1",
    valid_from: datetime = T0,
    valid_until: datetime | None = None,
    observed_at: datetime = T0,
    name: str = "get_user",
) -> dict[str, Any]:
    return {
        "id": entity_id,
        "version_id": version_id,
        "kind": "function",
        "name": name,
        "qualified_name": f"myrepo.{name}",
        "repo": "myrepo",
        "file_path": "src/users.py",
        "valid_from": valid_from.isoformat(),
        "valid_until": valid_until.isoformat() if valid_until else None,
        "observed_at": observed_at.isoformat(),
        "change_source": "initial_ingestion",
        "commit_sha": None,
        "agent_id": None,
        "superseded_by": None,
    }


# ── _inject_at_filter ─────────────────────────────────────────────────────────


class TestInjectAtFilter:
    def test_replaces_valid_until_is_null(self) -> None:
        q = "MATCH (e:Entity) WHERE e.valid_until IS NULL RETURN e"
        result = _inject_at_filter(q, "2026-01-02T00:00:00+00:00")
        assert "e.valid_from <= $__at" in result
        assert "e.valid_until IS NULL OR e.valid_until > $__at" in result
        assert "e.valid_until IS NULL RETURN" not in result

    def test_inserts_before_return_when_no_where(self) -> None:
        q = "MATCH (e:Entity) RETURN e"
        result = _inject_at_filter(q, "2026-01-02T00:00:00+00:00")
        assert "WHERE" in result
        assert result.index("WHERE") < result.index("RETURN")

    def test_appends_to_existing_where(self) -> None:
        q = "MATCH (e:Entity) WHERE e.repo = 'x' RETURN e"
        result = _inject_at_filter(q, "2026-01-02T00:00:00+00:00")
        assert "e.valid_from <= $__at" in result
        assert "e.repo = 'x'" in result


# ── _compute_changed_fields ───────────────────────────────────────────────────


class TestComputeChangedFields:
    def test_identical_entities_no_changes(self) -> None:
        row_v1 = entity_row("e-1", "v-1", name="foo")
        row_v2 = entity_row("e-1", "v-2", name="foo")
        from tractable.graph.temporal_graph import _row_to_entity

        e1 = _row_to_entity(row_v1)
        e2 = _row_to_entity(row_v2)
        assert _compute_changed_fields(e1, e2) == []

    def test_name_change_detected(self) -> None:
        from tractable.graph.temporal_graph import _row_to_entity

        e1 = _row_to_entity(entity_row(name="old_name"))
        e2 = _row_to_entity(entity_row(name="new_name"))
        assert "name" in _compute_changed_fields(e1, e2)

    def test_file_path_change_detected(self) -> None:
        from tractable.graph.temporal_graph import _row_to_entity

        row1 = entity_row()
        row2 = entity_row()
        row2["file_path"] = "src/new.py"
        e1 = _row_to_entity(row1)
        e2 = _row_to_entity(row2)
        assert "file_path" in _compute_changed_fields(e1, e2)


# ── get_entity_at ─────────────────────────────────────────────────────────────


class TestGetEntityAt:
    @pytest.mark.asyncio
    async def test_returns_entity_when_found(self) -> None:
        row = entity_row(valid_from=T0, valid_until=T2)
        graph, _ = make_graph(default_return=[row])
        result = await graph.get_entity_at("e-1", T1)
        assert result is not None
        assert result.id == "e-1"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self) -> None:
        graph, _ = make_graph(default_return=[])
        result = await graph.get_entity_at("missing", T1)
        assert result is None

    @pytest.mark.asyncio
    async def test_passes_at_param(self) -> None:
        graph, mock = make_graph(default_return=[entity_row()])
        at_time = T1
        await graph.get_entity_at("e-1", at_time)
        call_args = mock.execute.call_args
        params = call_args[0][1]
        assert params["at"] == at_time.isoformat()
        assert params["id"] == "e-1"

    @pytest.mark.asyncio
    async def test_filter_in_cypher(self) -> None:
        graph, mock = make_graph(default_return=[entity_row()])
        await graph.get_entity_at("e-1", T1)
        cypher: str = mock.execute.call_args[0][0]
        assert "valid_from <= $at" in cypher
        assert "valid_until > $at" in cypher


# ── get_entity_history ────────────────────────────────────────────────────────


class TestGetEntityHistory:
    @pytest.mark.asyncio
    async def test_returns_both_versions(self) -> None:
        v1 = entity_row("e-1", "v-1", valid_from=T0, valid_until=T1)
        v2 = entity_row("e-1", "v-2", valid_from=T1, valid_until=None, observed_at=T1)
        graph, _ = make_graph(default_return=[v1, v2])
        result = await graph.get_entity_history("e-1")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_passes_entity_id(self) -> None:
        graph, mock = make_graph(default_return=[])
        await graph.get_entity_history("e-42")
        params = mock.execute.call_args[0][1]
        assert params["id"] == "e-42"

    @pytest.mark.asyncio
    async def test_since_filter_passed(self) -> None:
        graph, mock = make_graph(default_return=[])
        await graph.get_entity_history("e-1", since=T1)
        cypher: str = mock.execute.call_args[0][0]
        params = mock.execute.call_args[0][1]
        assert "valid_from >= $since" in cypher
        assert params["since"] == T1.isoformat()

    @pytest.mark.asyncio
    async def test_until_filter_passed(self) -> None:
        graph, mock = make_graph(default_return=[])
        await graph.get_entity_history("e-1", until=T2)
        cypher: str = mock.execute.call_args[0][0]
        params = mock.execute.call_args[0][1]
        assert "valid_from <= $until" in cypher
        assert params["until"] == T2.isoformat()


# ── query_at ──────────────────────────────────────────────────────────────────


class TestQueryAt:
    @pytest.mark.asyncio
    async def test_injects_temporal_filter(self) -> None:
        graph, mock = make_graph(default_return=[])
        await graph.query_at("MATCH (e:Entity) RETURN e", T1)
        cypher: str = mock.execute.call_args[0][0]
        assert "e.valid_from <= $__at" in cypher

    @pytest.mark.asyncio
    async def test_passes_at_as_param(self) -> None:
        graph, mock = make_graph(default_return=[])
        await graph.query_at("MATCH (e:Entity) RETURN e", T1)
        params: dict[str, Any] = mock.execute.call_args[0][1]
        assert params["__at"] == T1.isoformat()

    @pytest.mark.asyncio
    async def test_merges_additional_params(self) -> None:
        graph, mock = make_graph(default_return=[])
        await graph.query_at("MATCH (e:Entity) RETURN e", T1, {"repo": "myrepo"})
        params: dict[str, Any] = mock.execute.call_args[0][1]
        assert params["repo"] == "myrepo"
        assert params["__at"] == T1.isoformat()


# ── get_changes_since ─────────────────────────────────────────────────────────


class TestGetChangesSince:
    @pytest.mark.asyncio
    async def test_added_entity_no_prior_version(self) -> None:
        """Entity in Q1, empty Q2 (no prior), empty Q3 → entities_added."""
        row = entity_row("e-1", "v-1", valid_from=T1, observed_at=T1)
        # Q1 returns the new entity; Q2 returns no prior; Q3 returns nothing
        graph, _ = make_graph(side_effects=[[row], [], []])
        result = await graph.get_changes_since(T0)
        assert len(result.entities_added) == 1
        assert result.entities_added[0].id == "e-1"
        assert result.entities_modified == []
        assert result.entities_removed == []

    @pytest.mark.asyncio
    async def test_modified_entity_with_prior_version(self) -> None:
        """Entity in Q1 AND Q2 (prior exists) → entities_modified."""
        current = entity_row("e-1", "v-2", valid_from=T1, observed_at=T1)
        prior = entity_row("e-1", "v-1", valid_from=T0, valid_until=T1, observed_at=T0)
        graph, _ = make_graph(side_effects=[[current], [prior], []])
        result = await graph.get_changes_since(T1 - timedelta(minutes=1))
        assert len(result.entities_modified) == 1
        assert result.entities_modified[0].entity_id == "e-1"
        assert result.entities_modified[0].previous_version.version_id == "v-1"
        assert result.entities_modified[0].current_version.version_id == "v-2"
        assert result.entities_added == []

    @pytest.mark.asyncio
    async def test_removed_entity_not_in_current(self) -> None:
        """Entity in Q3 (closed) but NOT in Q1 (no current version) → removed.

        When Q1 is empty, Q2 (prior version lookup) is skipped — so only 2
        execute calls happen: Q1 then Q3.
        """
        closed = entity_row("e-1", "v-1", valid_from=T0, valid_until=T1, observed_at=T0)
        # Q1 returns nothing (Q2 skipped); Q3 returns the closed entity
        graph, _ = make_graph(side_effects=[[], [closed]])
        result = await graph.get_changes_since(T0)
        assert len(result.entities_removed) == 1
        assert result.entities_removed[0].id == "e-1"

    @pytest.mark.asyncio
    async def test_returns_empty_changeset_on_empty_graph(self) -> None:
        graph, _ = make_graph(side_effects=[[], [], []])
        result = await graph.get_changes_since(T0)
        assert result.is_empty

    @pytest.mark.asyncio
    async def test_changeset_time_range_start_equals_since(self) -> None:
        graph, _ = make_graph(side_effects=[[], [], []])
        result = await graph.get_changes_since(T1)
        assert result.time_range_start == T1

    @pytest.mark.asyncio
    async def test_repo_filter_passed_to_query(self) -> None:
        graph, mock = make_graph(side_effects=[[], [], []])
        await graph.get_changes_since(T0, repo="myrepo")
        first_call_params = mock.execute.call_args_list[0][0][1]
        assert first_call_params.get("repo") == "myrepo"

    @pytest.mark.asyncio
    async def test_commits_collected(self) -> None:
        row = entity_row("e-1", "v-1", valid_from=T1, observed_at=T1)
        row["commit_sha"] = "abc123"
        graph, _ = make_graph(side_effects=[[row], [], []])
        result = await graph.get_changes_since(T0)
        assert "abc123" in result.commits

    @pytest.mark.asyncio
    async def test_changed_fields_populated(self) -> None:
        current = entity_row("e-1", "v-2", valid_from=T1, observed_at=T1, name="new_fn")
        prior = entity_row(
            "e-1", "v-1", valid_from=T0, valid_until=T1, observed_at=T0, name="old_fn"
        )
        graph, _ = make_graph(side_effects=[[current], [prior], []])
        result = await graph.get_changes_since(T0)
        assert "name" in result.entities_modified[0].changed_fields


# ── get_changes_between ────────────────────────────────────────────────────────


class TestGetChangesBetween:
    @pytest.mark.asyncio
    async def test_returns_changeset(self) -> None:
        graph, _ = make_graph(side_effects=[[], [], []])
        result = await graph.get_changes_between(T0, T2)
        assert result.time_range_start == T0
        assert result.time_range_end == T2


# ── diff_graph ────────────────────────────────────────────────────────────────


class TestDiffGraph:
    @pytest.mark.asyncio
    async def test_added_entity_between_snapshots(self) -> None:
        """Entity present at T1 but not T0 → appears in added_entities."""
        row = entity_row("e-1", "v-1", valid_from=T0)
        # Q for time_a returns nothing; Q for time_b returns the entity
        graph, _ = make_graph(side_effects=[[], [row]])
        result = await graph.diff_graph(T0 - timedelta(days=1), T1)
        assert len(result.added_entities) == 1
        assert result.added_entities[0].id == "e-1"

    @pytest.mark.asyncio
    async def test_removed_entity_between_snapshots(self) -> None:
        """Entity present at T0 but not T1 → appears in removed_entities."""
        row = entity_row("e-1", "v-1", valid_from=T0, valid_until=T1)
        graph, _ = make_graph(side_effects=[[row], []])
        result = await graph.diff_graph(T0, T2)
        assert len(result.removed_entities) == 1

    @pytest.mark.asyncio
    async def test_modified_entity_different_version(self) -> None:
        """Same entity id but different version_id → modified."""
        row_a = entity_row("e-1", "v-1", valid_from=T0)
        row_b = entity_row("e-1", "v-2", valid_from=T1, observed_at=T1)
        graph, _ = make_graph(side_effects=[[row_a], [row_b]])
        result = await graph.diff_graph(T0, T2)
        assert len(result.modified_entities) == 1
        assert result.modified_entities[0].entity_id == "e-1"

    @pytest.mark.asyncio
    async def test_empty_diff_when_no_changes(self) -> None:
        row = entity_row("e-1", "v-1", valid_from=T0)
        graph, _ = make_graph(side_effects=[[row], [row]])
        result = await graph.diff_graph(T0, T1)
        assert result.added_entities == []
        assert result.removed_entities == []
        assert result.modified_entities == []

    @pytest.mark.asyncio
    async def test_repos_affected_populated(self) -> None:
        row_b = entity_row("e-1", "v-1", valid_from=T0)
        graph, _ = make_graph(side_effects=[[], [row_b]])
        result = await graph.diff_graph(T0, T1)
        assert "myrepo" in result.repos_affected
