# ADR-001: Graph Database Selection — FalkorDB vs Alternatives

**Date:** 2026-03-25
**Status:** Accepted
**Deciders:** Tractable core team

---

## Context

Tractable uses a temporal property graph to store code entities (functions, classes, files, imports) with version history. All graph nodes carry `valid_from`, `valid_until`, `observed_at`, and `commit_sha` to support time-travel queries.

The current backend is **FalkorDB**, a community-maintained fork of RedisGraph (deprecated by Redis Ltd in 2023). FalkorDB is Cypher-compatible and runs embedded in the existing Docker Compose stack alongside Redis. This was the right call for MVP: zero-config self-hosting, no additional service, and the `TemporalCodeGraph` Protocol isolates the agent from the backend implementation.

The risk flagged in Phase 2 planning: FalkorDB's long-term maintenance trajectory is uncertain. The project has a small contributor base (~20 active contributors as of March 2026), no commercial sponsor, and no SLA. A Phase 4 requirement—exposing the graph via an MCP server—intensifies the need for confidence in graph DB stability before investing in that layer.

This decision record evaluates four options and recommends one for Phase 4+.

---

## Options Evaluated

| # | Option | Description |
|---|---|---|
| A | **Stay on FalkorDB** | Continue using the current backend; monitor project health |
| B | **Migrate to Neo4j Community** | Self-hosted open-source Neo4j; free tier, mature |
| C | **Migrate to Neo4j AuraDB** | Neo4j-managed cloud service; no ops burden |
| D | **Migrate to Apache AGE** | PostgreSQL graph extension; eliminates a separate service entirely |

---

## Decision Criteria

| Criterion | Weight | Notes |
|---|---|---|
| Long-term maintenance risk | High | Sustainability of the project/vendor |
| Temporal Cypher query support | High | Queries use `valid_from`/`valid_until` predicates; must be expressive |
| Operational complexity | Medium | Dev and production deployment burden |
| Cost at MVP scale (<10 repos) | Low | All options are free or near-free at this scale |
| Cost at Phase 7 scale (1000s of agents) | Medium | Matters for planning; not a gate for Phase 4 |
| Protocol compatibility | High | Must satisfy `TemporalCodeGraph` interface with no changes to agent logic |

---

## Option Analysis

**Option A — Stay on FalkorDB.** Zero migration cost; proven to work with the current schema and temporal query patterns. The risk is maintenance trajectory: the most recent FalkorDB release (v4.x) lags behind upstream Cypher spec features needed for recursive path queries in Phase 5+. The project has 20 active contributors vs Neo4j's 200+. If the project stagnates, a mid-Phase migration under time pressure is worse than a planned one now.

**Option B — Neo4j Community (self-hosted).** Neo4j is the reference implementation of Cypher and has 10+ years of production deployments at scale. Community edition supports all Cypher features used by the current temporal schema. Limitation: no native clustering in Community edition (clustering requires Enterprise). At MVP and Phase 4 scale this is not a constraint. Docker image is available; adding it to `docker-compose.yml` is a one-day effort. The `TemporalCodeGraph` Protocol means the migration is scoped to `tractable/graph/client.py` and `tractable/graph/temporal_graph.py` — no agent logic changes.

**Option C — Neo4j AuraDB (managed).** Eliminates all operational burden — no Docker container, no schema init scripts, no backup management. The free tier supports 200k nodes, which is sufficient for Phase 4 testing (~50k nodes per mid-size repo). Cost at Phase 7 scale requires evaluation; AuraDB pricing scales with node/relationship count. This option is not appropriate as the Phase 4 default because it introduces an external dependency that breaks offline/air-gapped dev environments.

**Option D — Apache AGE (PostgreSQL graph extension).** Eliminates the graph service entirely — agent state and graph live in one PostgreSQL instance. However, AGE's Cypher support is partial: recursive path queries (`shortestPath`, variable-length relationships with predicates) are not fully supported as of v1.5. The temporal query pattern `MATCH (f:File) WHERE f.valid_until IS NULL AND f.valid_from <= $t` is supported, but more complex traversals needed in Phase 5+ are not. Migration would require validating every query in `tractable/graph/`, which is higher-risk than a FalkorDB → Neo4j swap where Cypher is fully compatible.

---

## Recommendation

**Migrate to Neo4j Community (self-hosted) in Phase 4, Milestone 4.1.**

Supporting data points:
1. **Maintenance trajectory:** Neo4j Community's GitHub repository has 580+ contributors and a commercially-backed release cadence (quarterly minor releases, LTS policy). FalkorDB's last release was 14 months ago with no published roadmap. The risk of FalkorDB stagnating during the Phase 5–7 build is material.
2. **Cypher compatibility:** Neo4j Community passes all Cypher TCK tests. The `TemporalCodeGraph` Protocol's existing query surface (8 methods, all standard Cypher) requires no changes — migration is a client swap, not a query rewrite. A spike on `tractable/graph/client.py` confirmed that the `neo4j` Python driver's async API matches the current FalkorDB client interface within a single adapter class.

The migration fits in one Phase 4 milestone (estimated 3–4 days: driver swap, schema re-init, integration test pass). It does not require its own phase.

AuraDB should be revisited for Phase 7 production deployments where managed infrastructure is preferred.

---

## Consequences

**If adopted:**
- `tractable/graph/client.py` is replaced with a Neo4j async driver adapter.
- `deploy/docker-compose.yml` replaces the `falkordb` service with the official `neo4j:5-community` image.
- `deploy/init/falkordb-schema.cypher` is renamed and validated against Neo4j syntax (no changes expected — schema uses standard Cypher DDL).
- All integration tests in `tests/integration/test_graph_*.py` pass without modification (Protocol contract is unchanged).
- FalkorDB dependency is removed from `pyproject.toml`.

**What stays the same:**
- Agent logic — zero changes.
- `TemporalCodeGraph` Protocol — unchanged.
- Temporal schema and all query patterns — unchanged.
- Dev workflow — `docker compose up` still starts the full stack.

**Migration risks:**
- Neo4j Community has no Lua scripting (not used). No other FalkorDB-specific features are in use.
- The `neo4j` Python driver requires `bolt://` URIs; the current `GRAPH_URL` env var will need renaming (`NEO4J_URL`). One-line change in `tractable/graph/client.py` and `.env.example`.
