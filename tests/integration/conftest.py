"""Shared fixtures for integration tests.

These fixtures connect to the live Docker Compose stack. Start the stack with:
    docker compose -f deploy/docker-compose.yml up -d

Environment variables:
    FALKORDB_HOST     FalkorDB hostname    (default: localhost)
    FALKORDB_PORT     FalkorDB port        (default: 6380)
    DATABASE_URL      PostgreSQL DSN       (default: see below)
    GITHUB_TEST_TOKEN GitHub PAT with repo scope (required for GitHub tests)
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

import pytest

# ── Connection defaults ───────────────────────────────────────────────────────

_FALKORDB_HOST = os.environ.get("FALKORDB_HOST", "localhost")
_FALKORDB_PORT = int(os.environ.get("FALKORDB_PORT", "6380"))
_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://tractable:tractable_dev@localhost:5433/tractable",
)

# Graph name used for all integration tests — cleaned between tests
_INTEGRATION_GRAPH = "tractable_integration_test"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
async def graph() -> AsyncGenerator[object, None]:
    """Live FalkorDBTemporalGraph connected to the integration test graph.

    Deletes all Entity nodes in the graph after each test to ensure isolation.
    """
    from tractable.graph.client import FalkorDBClient
    from tractable.graph.temporal_graph import FalkorDBTemporalGraph

    client = FalkorDBClient(
        host=_FALKORDB_HOST,
        port=_FALKORDB_PORT,
        graph_name=_INTEGRATION_GRAPH,
    )
    g = FalkorDBTemporalGraph(client)
    yield g
    # Teardown: remove all entities and edges created during this test
    try:
        await client.execute_write("MATCH (n) DELETE n", {})
    finally:
        await client.close()


@pytest.fixture()
def state_store() -> object:
    """Live PostgreSQLAgentStateStore connected to the integration database."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from tractable.state.store import PostgreSQLAgentStateStore

    engine = create_async_engine(_DATABASE_URL, pool_pre_ping=True)
    factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        engine, expire_on_commit=False
    )
    return PostgreSQLAgentStateStore(factory)


@pytest.fixture()
def github_token() -> str:
    """Return the GITHUB_TEST_TOKEN or skip the test if it is not set."""
    token = os.environ.get("GITHUB_TEST_TOKEN")
    if not token:
        pytest.skip("GITHUB_TEST_TOKEN environment variable not set — skipping GitHub test")
    return token
