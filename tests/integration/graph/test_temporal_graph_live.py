"""Integration tests for FalkorDBTemporalGraph against a live FalkorDB.

Requires the Docker Compose stack:
  docker compose -f deploy/docker-compose.yml up -d falkordb

Run with:
  uv run pytest tests/integration/graph/test_temporal_graph_live.py
"""

from __future__ import annotations

import uuid

import pytest

from tractable.graph.client import FalkorDBClient
from tractable.graph.temporal_graph import FalkorDBTemporalGraph
from tractable.types.enums import ChangeSource
from tractable.types.temporal import TemporalMutation

# Graph name isolated from main and client-level tests
_GRAPH = "tractable_tg_test"


@pytest.fixture()
async def graph() -> FalkorDBTemporalGraph:
    client = FalkorDBClient(host="localhost", port=6380, graph_name=_GRAPH)
    return FalkorDBTemporalGraph(client)


def _entity_payload(entity_id: str, version_id: str = "v1") -> dict[str, object]:
    return {
        "id": entity_id,
        "version_id": version_id,
        "kind": "function",
        "name": "test_fn",
        "qualified_name": f"testrepo.{entity_id}",
        "repo": "testrepo",
        "file_path": "src/test.py",
    }


# ── AC1 — create_entity, then get_current_entity returns is_current True ──────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_entity_is_current(graph: FalkorDBTemporalGraph) -> None:
    eid = f"tg-{uuid.uuid4()}"
    mutation = TemporalMutation(
        operation="create_entity",
        payload=_entity_payload(eid),
    )
    result = await graph.apply_mutations([mutation], ChangeSource.INITIAL_INGESTION)
    assert result.errors == [], result.errors

    entity = await graph.get_current_entity(eid)
    assert entity is not None, "Entity not found after create"
    assert entity.id == eid
    assert entity.is_current is True

    # Cleanup
    await graph._client.execute_write("MATCH (e:Entity {id: $id}) DELETE e", {"id": eid})


# ── AC2 — update_entity closes old version, new version is current ────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_update_entity_versioning(graph: FalkorDBTemporalGraph) -> None:
    eid = f"tg-{uuid.uuid4()}"
    vid1 = str(uuid.uuid4())
    vid2 = str(uuid.uuid4())

    # Create v1
    await graph.apply_mutations(
        [TemporalMutation(operation="create_entity", payload=_entity_payload(eid, vid1))],
        ChangeSource.INITIAL_INGESTION,
    )

    # Update to v2
    await graph.apply_mutations(
        [
            TemporalMutation(
                operation="update_entity",
                entity_id=eid,
                payload={**_entity_payload(eid, vid2), "name": "updated_fn"},
            )
        ],
        ChangeSource.HUMAN_COMMIT,
    )

    # Current entity is v2
    current = await graph.get_current_entity(eid)
    assert current is not None
    assert current.is_current is True

    # v1 has valid_until set (closed)
    rows = await graph._client.execute(
        "MATCH (e:Entity {id: $id, version_id: $vid}) RETURN e.valid_until AS vu",
        {"id": eid, "vid": vid1},
    )
    assert len(rows) == 1
    assert rows[0]["vu"] is not None, "v1 valid_until should be set after update"

    # Cleanup
    await graph._client.execute_write("MATCH (e:Entity {id: $id}) DELETE e", {"id": eid})


# ── AC3 — delete_entity, get_current_entity returns None ─────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_entity_returns_none(graph: FalkorDBTemporalGraph) -> None:
    eid = f"tg-{uuid.uuid4()}"

    await graph.apply_mutations(
        [TemporalMutation(operation="create_entity", payload=_entity_payload(eid))],
        ChangeSource.INITIAL_INGESTION,
    )

    # Confirm it exists
    assert await graph.get_current_entity(eid) is not None

    # Delete
    await graph.apply_mutations(
        [TemporalMutation(operation="delete_entity", entity_id=eid)],
        ChangeSource.HUMAN_COMMIT,
    )

    # Now gone from current view
    assert await graph.get_current_entity(eid) is None

    # Cleanup (the node still exists with valid_until set)
    await graph._client.execute_write("MATCH (e:Entity {id: $id}) DELETE e", {"id": eid})


# ── AC4 — impact_analysis_current on graph with 5 connected entities ─────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_impact_analysis_current_five_entities(graph: FalkorDBTemporalGraph) -> None:
    prefix = f"ia-{uuid.uuid4()}"
    ids = [f"{prefix}-{i}" for i in range(5)]

    # Create 5 entities
    mutations = [
        TemporalMutation(operation="create_entity", payload=_entity_payload(eid)) for eid in ids
    ]
    await graph.apply_mutations(mutations, ChangeSource.INITIAL_INGESTION)

    # Connect: 0→1, 1→2, 2→3, 3→4
    for i in range(4):
        await graph.apply_mutations(
            [
                TemporalMutation(
                    operation="create_edge",
                    payload={
                        "source_entity_id": ids[i],
                        "target_entity_id": ids[i + 1],
                        "confidence": 0.9,
                        "relationship": "CALLS",
                    },
                )
            ],
            ChangeSource.INITIAL_INGESTION,
        )

    # Run impact analysis from the root
    report = await graph.impact_analysis_current([ids[0]], depth=2, min_confidence=0.5)
    assert len(report.directly_affected) >= 1, "Should have at least 1 direct neighbor"

    # Cleanup
    for eid in ids:
        await graph._client.execute_write("MATCH (e:Entity {id: $id}) DELETE e", {"id": eid})
