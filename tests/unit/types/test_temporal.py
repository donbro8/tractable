"""Unit tests for tractable/types/temporal.py."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tractable.types.enums import ChangeRelevance, ChangeSource
from tractable.types.temporal import (
    ChangeNotification,
    ChangeSet,
    EntityModification,
    GraphDiff,
    TemporalEdge,
    TemporalGraphEntity,
    TemporalMetadata,
    TemporalMutation,
    TemporalMutationResult,
)

T0 = datetime(2026, 3, 17, 10, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 3, 17, 11, 0, 0, tzinfo=UTC)
T2 = datetime(2026, 3, 17, 12, 0, 0, tzinfo=UTC)


def _meta(
    valid_until: datetime | None = None,
    source: ChangeSource = ChangeSource.INITIAL_INGESTION,
) -> TemporalMetadata:
    return TemporalMetadata(
        valid_from=T0,
        valid_until=valid_until,
        observed_at=T0,
        change_source=source,
    )


def _entity(
    eid: str = "e1",
    repo: str = "repo-a",
    valid_until: datetime | None = None,
) -> TemporalGraphEntity:
    return TemporalGraphEntity(
        id=eid,
        version_id=f"{eid}-v1",
        kind="function",
        name="foo",
        qualified_name=f"{repo}.foo",
        repo=repo,
        file_path="src/foo.py",
        temporal=_meta(valid_until=valid_until),
    )


def _edge(valid_until: datetime | None = None) -> TemporalEdge:
    return TemporalEdge(
        edge_id="edge1",
        version_id="edge1-v1",
        source_entity_id="e1",
        target_entity_id="e2",
        relationship="CALLS",
        confidence=0.9,
        resolution="deterministic",
        temporal=_meta(valid_until=valid_until),
    )


# ── TemporalMetadata ───────────────────────────────────────────────────


def test_temporal_metadata_current() -> None:
    m = _meta()
    assert m.valid_until is None
    assert m.superseded_by is None
    assert m.agent_id is None


def test_temporal_metadata_historical() -> None:
    m = _meta(valid_until=T1)
    assert m.valid_until == T1


# ── TemporalGraphEntity — is_current ──────────────────────────────────


def test_temporal_entity_is_current_true() -> None:
    e = _entity(valid_until=None)
    assert e.is_current is True


def test_temporal_entity_is_current_false() -> None:
    e = _entity(valid_until=T1)
    assert e.is_current is False


def test_temporal_entity_properties_default() -> None:
    e = _entity()
    assert e.properties == {}
    assert e.line_start is None
    assert e.line_end is None


# ── TemporalEdge — is_current ─────────────────────────────────────────


def test_temporal_edge_is_current_true() -> None:
    assert _edge(valid_until=None).is_current is True


def test_temporal_edge_is_current_false() -> None:
    assert _edge(valid_until=T1).is_current is False


# ── ChangeSet — is_empty and summary ──────────────────────────────────


def test_changeset_is_empty_when_no_changes() -> None:
    cs = ChangeSet(time_range_start=T0, time_range_end=T1)
    assert cs.is_empty is True


def test_changeset_is_not_empty_with_added_entity() -> None:
    cs = ChangeSet(time_range_start=T0, time_range_end=T1, entities_added=[_entity()])
    assert cs.is_empty is False


def test_changeset_summary_no_changes() -> None:
    cs = ChangeSet(time_range_start=T0, time_range_end=T1)
    assert cs.summary == "No changes"


def test_changeset_summary_with_entities_added() -> None:
    cs = ChangeSet(
        time_range_start=T0,
        time_range_end=T1,
        entities_added=[_entity("e1"), _entity("e2")],
    )
    assert "2 entities added" in cs.summary


def test_changeset_summary_mixed() -> None:
    e1 = _entity("e1")
    e2 = _entity("e2")
    mod = EntityModification(
        entity_id="e3",
        previous_version=_entity("e3", valid_until=T1),
        current_version=_entity("e3"),
        changed_fields=["name"],
        change_description="renamed",
    )
    cs = ChangeSet(
        time_range_start=T0,
        time_range_end=T1,
        entities_added=[e1],
        entities_modified=[mod],
        entities_removed=[e2],
    )
    assert "1 entities added" in cs.summary
    assert "1 entities modified" in cs.summary
    assert "1 entities removed" in cs.summary


def test_changeset_defaults_independent() -> None:
    cs1 = ChangeSet(time_range_start=T0, time_range_end=T1)
    cs2 = ChangeSet(time_range_start=T0, time_range_end=T1)
    cs1.commits.append("abc")
    assert cs2.commits == []


# ── GraphDiff — for_repo ──────────────────────────────────────────────


def test_graph_diff_for_repo_filters_entities() -> None:
    diff = GraphDiff(
        time_a=T0,
        time_b=T1,
        added_entities=[_entity("e1", repo="repo-a"), _entity("e2", repo="repo-b")],
        repos_affected=["repo-a", "repo-b"],
    )
    filtered = diff.for_repo("repo-a")
    assert len(filtered.added_entities) == 1
    assert filtered.added_entities[0].repo == "repo-a"
    assert filtered.repos_affected == ["repo-a"]


def test_graph_diff_for_repo_returns_graph_diff() -> None:
    diff = GraphDiff(time_a=T0, time_b=T1)
    result = diff.for_repo("any-repo")
    assert isinstance(result, GraphDiff)


# ── TemporalMutation ───────────────────────────────────────────────────


def test_temporal_mutation_valid_operations() -> None:
    for op in (
        "create_entity",
        "update_entity",
        "delete_entity",
        "create_edge",
        "update_edge",
        "delete_edge",
    ):
        m = TemporalMutation(operation=op)  # type: ignore[arg-type]
        assert m.operation == op


def test_temporal_mutation_invalid_operation() -> None:
    with pytest.raises(ValidationError):
        TemporalMutation(operation="nuke_everything")  # type: ignore[arg-type]


def test_temporal_mutation_defaults() -> None:
    m = TemporalMutation(operation="create_entity")
    assert m.entity_id is None
    assert m.edge_id is None
    assert m.payload == {}


# ── TemporalMutationResult ─────────────────────────────────────────────


def test_temporal_mutation_result_defaults() -> None:
    r = TemporalMutationResult(
        entities_created=1,
        entities_updated=0,
        entities_deleted=0,
        edges_created=0,
        edges_deleted=0,
        timestamp=T2,
    )
    assert r.errors == []


# ── ChangeNotification ─────────────────────────────────────────────────


def test_change_notification_instantiation() -> None:
    n = ChangeNotification(
        target_agent_id="agent-1",
        repo_name="my-repo",
        relevance=ChangeRelevance.DIRECT,
        change_summary="Two functions added",
        commit_sha="abc123",
        requires_action=True,
    )
    assert n.affected_entity_ids == []
    assert n.requires_action is True


def test_change_notification_invalid_relevance() -> None:
    with pytest.raises(ValidationError):
        ChangeNotification(
            target_agent_id="a",
            repo_name="r",
            relevance="unknown",  # type: ignore[arg-type]
            change_summary="x",
            commit_sha="sha",
            requires_action=False,
        )
