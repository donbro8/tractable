"""Integration tests for FalkorDBClient against a live FalkorDB instance.

Requires the Docker Compose stack:
  docker compose -f deploy/docker-compose.yml up -d falkordb

Run with:
  uv run pytest tests/integration/graph/test_client_live.py
"""

from __future__ import annotations

import pytest

from tractable.graph.client import FalkorDBClient


@pytest.fixture()
async def client() -> FalkorDBClient:
    """Return a FalkorDBClient pointed at the local Docker Compose instance."""
    return FalkorDBClient(host="localhost", port=6380, graph_name="tractable_test")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ping_returns_true(client: FalkorDBClient) -> None:
    """FalkorDB must be reachable and respond to PING."""
    assert await client.ping() is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_count_on_fresh_graph_is_zero(client: FalkorDBClient) -> None:
    """A fresh graph returns count(e) == 0."""
    rows = await client.execute("MATCH (e) RETURN count(e) AS cnt", {})
    assert rows == [{"cnt": 0}]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_write_and_read_entity(client: FalkorDBClient) -> None:
    """A created entity must be retrievable immediately."""
    # Write
    await client.execute_write(
        "CREATE (:Entity {id: 'live-test-1', valid_until: null, repo: 'test-repo'})",
        {},
    )
    # Read back
    rows = await client.execute(
        "MATCH (e:Entity {id: 'live-test-1'}) RETURN e.id AS id, e.repo AS repo",
        {},
    )
    assert len(rows) == 1
    assert rows[0]["id"] == "live-test-1"
    assert rows[0]["repo"] == "test-repo"

    # Cleanup
    await client.execute_write("MATCH (e:Entity {id: 'live-test-1'}) DELETE e", {})


@pytest.mark.integration
@pytest.mark.asyncio
async def test_close_entity_version(client: FalkorDBClient) -> None:
    """Setting valid_until on a node must persist correctly."""
    await client.execute_write(
        "CREATE (:Entity {id: 'live-ver-1', version_id: 'v1', valid_until: null})",
        {},
    )
    await client.execute_write(
        "MATCH (e:Entity {id: 'live-ver-1', version_id: 'v1'}) "
        "SET e.valid_until = '2026-03-19T00:00:00'",
        {},
    )
    rows = await client.execute(
        "MATCH (e:Entity {id: 'live-ver-1'}) RETURN e.valid_until AS vu",
        {},
    )
    assert len(rows) == 1
    assert rows[0]["vu"] is not None

    # Cleanup
    await client.execute_write("MATCH (e:Entity {id: 'live-ver-1'}) DELETE e", {})
