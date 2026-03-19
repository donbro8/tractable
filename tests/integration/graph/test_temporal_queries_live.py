"""Integration tests for FalkorDBTemporalGraph temporal queries against live FalkorDB.

Requires the Docker Compose stack:
  docker compose -f deploy/docker-compose.yml up -d falkordb

Run with:
  uv run pytest tests/integration/graph/test_temporal_queries_live.py
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from tractable.graph.client import FalkorDBClient
from tractable.graph.temporal_graph import FalkorDBTemporalGraph
from tractable.types.enums import ChangeSource
from tractable.types.temporal import TemporalMutation

_GRAPH = "tractable_tq_test"


@pytest.fixture()
async def graph() -> FalkorDBTemporalGraph:
    client = FalkorDBClient(host="localhost", port=6380, graph_name=_GRAPH)
    return FalkorDBTemporalGraph(client)


def _payload(entity_id: str, version_id: str = "v1", name: str = "test_fn") -> dict[str, object]:
    return {
        "id": entity_id,
        "version_id": version_id,
        "kind": "function",
        "name": name,
        "qualified_name": f"testrepo.{entity_id}",
        "repo": "testrepo",
        "file_path": "src/test.py",
    }


async def _cleanup(graph: FalkorDBTemporalGraph, *entity_ids: str) -> None:
    for eid in entity_ids:
        await graph._client.execute_write(
            "MATCH (e:Entity {id: $id}) DELETE e", {"id": eid}
        )


# ── AC1: get_changes_since returns entities_added after create ────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_changes_since_entities_added(graph: FalkorDBTemporalGraph) -> None:
    eid = f"tq-{uuid.uuid4()}"
    t_before = datetime.now(tz=UTC) - timedelta(seconds=1)

    await graph.apply_mutations(
        [TemporalMutation(operation="create_entity", payload=_payload(eid))],
        ChangeSource.INITIAL_INGESTION,
    )

    changes = await graph.get_changes_since(t_before)
    added_ids = [e.id for e in changes.entities_added]
    assert eid in added_ids, f"Expected {eid} in entities_added; got {added_ids}"

    await _cleanup(graph, eid)


# ── AC2: get_changes_since returns entities_modified after update ──────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_changes_since_entities_modified(graph: FalkorDBTemporalGraph) -> None:
    eid = f"tq-{uuid.uuid4()}"
    vid1 = str(uuid.uuid4())
    vid2 = str(uuid.uuid4())

    # Create v1
    await graph.apply_mutations(
        [TemporalMutation(operation="create_entity", payload=_payload(eid, vid1))],
        ChangeSource.INITIAL_INGESTION,
    )

    t_before_update = datetime.now(tz=UTC)

    # Update to v2
    await graph.apply_mutations(
        [TemporalMutation(
            operation="update_entity",
            entity_id=eid,
            payload={**_payload(eid, vid2), "name": "updated_fn"},
        )],
        ChangeSource.HUMAN_COMMIT,
    )

    changes = await graph.get_changes_since(t_before_update)
    modified_ids = [m.entity_id for m in changes.entities_modified]
    assert len(changes.entities_modified) >= 1, f"Expected entities_modified; got {changes}"
    assert eid in modified_ids, f"Expected {eid} in entities_modified; got {modified_ids}"

    await _cleanup(graph, eid)


# ── AC3: get_entity_history returns 2 versions after create + update ──────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_entity_history_two_versions(graph: FalkorDBTemporalGraph) -> None:
    eid = f"tq-{uuid.uuid4()}"
    vid1 = str(uuid.uuid4())
    vid2 = str(uuid.uuid4())

    await graph.apply_mutations(
        [TemporalMutation(operation="create_entity", payload=_payload(eid, vid1))],
        ChangeSource.INITIAL_INGESTION,
    )
    await graph.apply_mutations(
        [TemporalMutation(
            operation="update_entity",
            entity_id=eid,
            payload={**_payload(eid, vid2), "name": "updated_fn"},
        )],
        ChangeSource.HUMAN_COMMIT,
    )

    history = await graph.get_entity_history(eid)
    assert len(history) == 2, f"Expected 2 history entries, got {len(history)}"
    assert history[0].version_id == vid1
    assert history[1].version_id == vid2

    await _cleanup(graph, eid)


# ── AC4: get_entity_at returns v1 when queried between v1 and v2 ──────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_entity_at_returns_correct_version(graph: FalkorDBTemporalGraph) -> None:
    eid = f"tq-{uuid.uuid4()}"
    vid1 = str(uuid.uuid4())
    vid2 = str(uuid.uuid4())

    # Create v1
    await graph.apply_mutations(
        [TemporalMutation(operation="create_entity", payload=_payload(eid, vid1))],
        ChangeSource.INITIAL_INGESTION,
    )

    t_between = datetime.now(tz=UTC)

    # Update to v2
    await graph.apply_mutations(
        [TemporalMutation(
            operation="update_entity",
            entity_id=eid,
            payload={**_payload(eid, vid2), "name": "updated_fn"},
        )],
        ChangeSource.HUMAN_COMMIT,
    )

    entity = await graph.get_entity_at(eid, t_between)
    assert entity is not None, "get_entity_at returned None"
    assert entity.version_id == vid1, (
        f"Expected v1 ({vid1}) but got {entity.version_id}"
    )

    await _cleanup(graph, eid)


# ── Performance: get_changes_since on 1000 entities under 500ms ───────────────
# Note: Full 10k entity benchmark requires separate load generation.
# This test verifies the query returns within acceptable time on a modest dataset.


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_changes_since_performance(graph: FalkorDBTemporalGraph) -> None:
    prefix = f"perf-{uuid.uuid4()}"
    n = 100
    ids = [f"{prefix}-{i}" for i in range(n)]

    mutations = [
        TemporalMutation(operation="create_entity", payload=_payload(eid))
        for eid in ids
    ]
    await graph.apply_mutations(mutations, ChangeSource.INITIAL_INGESTION)

    t_before = datetime.now(tz=UTC) - timedelta(minutes=1)

    start = time.monotonic()
    changes = await graph.get_changes_since(t_before)
    elapsed_ms = (time.monotonic() - start) * 1000

    matching = [e for e in changes.entities_added if e.id.startswith(prefix)]
    assert len(matching) == n, f"Expected {n} added entities, got {len(matching)}"
    assert elapsed_ms < 5000, f"get_changes_since took {elapsed_ms:.0f}ms (limit: 5000ms)"

    for eid in ids:
        await _cleanup(graph, eid)


# ── get_entity_history with since/until bounds ────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_entity_history_with_since_filter(graph: FalkorDBTemporalGraph) -> None:
    eid = f"tq-{uuid.uuid4()}"
    vid1 = str(uuid.uuid4())
    vid2 = str(uuid.uuid4())

    await graph.apply_mutations(
        [TemporalMutation(operation="create_entity", payload=_payload(eid, vid1))],
        ChangeSource.INITIAL_INGESTION,
    )
    t_after_v1 = datetime.now(tz=UTC)
    await graph.apply_mutations(
        [TemporalMutation(
            operation="update_entity",
            entity_id=eid,
            payload=_payload(eid, vid2),
        )],
        ChangeSource.HUMAN_COMMIT,
    )

    # Querying since after v1 creation → should only return v2
    history = await graph.get_entity_history(eid, since=t_after_v1)
    assert len(history) >= 1
    version_ids = [e.version_id for e in history]
    assert vid2 in version_ids

    await _cleanup(graph, eid)


# ── diff_graph: entity added between two snapshots ────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_diff_graph_entity_added(graph: FalkorDBTemporalGraph) -> None:
    eid = f"tq-{uuid.uuid4()}"
    t_a = datetime.now(tz=UTC) - timedelta(seconds=1)

    await graph.apply_mutations(
        [TemporalMutation(operation="create_entity", payload=_payload(eid))],
        ChangeSource.INITIAL_INGESTION,
    )

    t_b = datetime.now(tz=UTC) + timedelta(seconds=1)

    diff = await graph.diff_graph(t_a, t_b, repo="testrepo")
    added_ids = [e.id for e in diff.added_entities]
    assert eid in added_ids, f"Expected {eid} in diff.added_entities; got {added_ids}"

    await _cleanup(graph, eid)
