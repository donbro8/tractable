"""FalkorDB implementation of the TemporalCodeGraph protocol.

Implements current-state operations and apply_mutations (TASK-1.4.2).
Temporal query methods (get_changes_since, get_entity_history, etc.)
are stubbed here and implemented in TASK-1.4.3.

Sources:
- realtime-temporal-spec.py §B — TemporalCodeGraph Protocol (lines 144–264)
- realtime-temporal-spec.py §A — temporal mutation patterns (lines 827–843)
- realtime-temporal-spec.py §H — PHASE 1 additions, mutation logic (lines 927–930)
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from tractable.graph.client import FalkorDBClient
from tractable.types.enums import ChangeRisk, ChangeSource
from tractable.types.graph import GraphEntity, ImpactReport
from tractable.types.temporal import (
    ChangeSet,
    GraphDiff,
    TemporalGraphEntity,
    TemporalMetadata,
    TemporalMutation,
    TemporalMutationResult,
)

# ── Module-level helpers ──────────────────────────────────────────────────────


def _inject_current_filter(cypher: str) -> str:
    """Inject ``e.valid_until IS NULL`` into a Cypher query.

    Searches for an existing WHERE clause (before RETURN) and prepends the
    condition. If no WHERE exists, inserts one before RETURN. Falls back to
    appending at end of query if neither keyword is found.
    """
    upper = cypher.upper()
    where_pos = upper.find(" WHERE ")
    return_pos = upper.find(" RETURN ")

    if where_pos != -1 and (return_pos == -1 or where_pos < return_pos):
        insert = where_pos + len(" WHERE ")
        return cypher[:insert] + "e.valid_until IS NULL AND " + cypher[insert:]
    if return_pos != -1:
        return cypher[:return_pos] + " WHERE e.valid_until IS NULL" + cypher[return_pos:]
    return cypher + " WHERE e.valid_until IS NULL"


def _row_to_entity(row: dict[str, Any]) -> TemporalGraphEntity:
    """Convert a flat FalkorDB property row to a ``TemporalGraphEntity``."""
    valid_until_raw: Any = row.get("valid_until")
    valid_until: datetime | None = (
        datetime.fromisoformat(str(valid_until_raw)) if valid_until_raw else None
    )
    superseded_by_raw: Any = row.get("superseded_by")
    superseded_by: str | None = str(superseded_by_raw) if superseded_by_raw else None
    commit_sha_raw: Any = row.get("commit_sha")
    commit_sha: str | None = str(commit_sha_raw) if commit_sha_raw else None
    agent_id_raw: Any = row.get("agent_id")
    agent_id: str | None = str(agent_id_raw) if agent_id_raw else None

    return TemporalGraphEntity(
        id=str(row["id"]),
        version_id=str(row["version_id"]),
        kind=str(row["kind"]),
        name=str(row["name"]),
        qualified_name=str(row["qualified_name"]),
        repo=str(row["repo"]),
        file_path=str(row["file_path"]),
        temporal=TemporalMetadata(
            valid_from=datetime.fromisoformat(str(row["valid_from"])),
            valid_until=valid_until,
            observed_at=datetime.fromisoformat(str(row["observed_at"])),
            superseded_by=superseded_by,
            change_source=ChangeSource(str(row["change_source"])),
            commit_sha=commit_sha,
            agent_id=agent_id,
        ),
    )


# ── FalkorDBTemporalGraph ─────────────────────────────────────────────────────


class FalkorDBTemporalGraph:
    """FalkorDB-backed implementation of the TemporalCodeGraph protocol.

    Current-state queries (``valid_until IS NULL``) are the fast path used by
    agents during normal operation. Temporal/change-awareness methods are
    implemented in TASK-1.4.3 and currently raise ``NotImplementedError``.
    """

    def __init__(self, client: FalkorDBClient) -> None:
        self._client = client

    # ── Current-state queries (fast path) ────────────────────────────────────

    async def query_current(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> Sequence[dict[str, Any]]:
        """Execute a Cypher query scoped to current entities (valid_until IS NULL).

        Injects ``WHERE e.valid_until IS NULL`` into the query automatically.
        The query must use ``e`` as the entity node alias.
        """
        filtered = _inject_current_filter(cypher)
        return await self._client.execute(filtered, params or {})

    async def get_current_entity(
        self,
        entity_id: str,
    ) -> TemporalGraphEntity | None:
        """Return the current (live) version of an entity, or ``None``."""
        rows = await self._client.execute(
            "MATCH (e:Entity {id: $id}) "
            "WHERE e.valid_until IS NULL "
            "RETURN e.id AS id, e.version_id AS version_id, e.kind AS kind, "
            "e.name AS name, e.qualified_name AS qualified_name, "
            "e.repo AS repo, e.file_path AS file_path, "
            "e.valid_from AS valid_from, e.valid_until AS valid_until, "
            "e.observed_at AS observed_at, e.change_source AS change_source, "
            "e.commit_sha AS commit_sha, e.agent_id AS agent_id, "
            "e.superseded_by AS superseded_by",
            {"id": entity_id},
        )
        if not rows:
            return None
        return _row_to_entity(rows[0])

    async def impact_analysis_current(
        self,
        entity_ids: Sequence[str],
        depth: int = 3,
        min_confidence: float = 0.5,
    ) -> ImpactReport:
        """BFS impact analysis across current-state entities and RELATES edges.

        Traverses up to ``depth`` hops from ``entity_ids``, filtering to
        current entities (``valid_until IS NULL``) and edges above
        ``min_confidence``. Returns directly- and transitively-affected
        entities.
        """
        directly: list[GraphEntity] = []
        transitively: list[GraphEntity] = []
        affected_repos: set[str] = set()
        seen_ids: set[str] = set(entity_ids)
        frontier: list[str] = list(entity_ids)

        for hop in range(1, depth + 1):
            if not frontier:
                break
            next_frontier: list[str] = []
            for eid in frontier:
                rows = await self._client.execute(
                    "MATCH (s:Entity {id: $start_id})-[r:RELATES]->(t:Entity) "
                    "WHERE s.valid_until IS NULL AND t.valid_until IS NULL "
                    "AND r.valid_until IS NULL "
                    "RETURN t.id AS id, t.kind AS kind, t.name AS name, "
                    "t.repo AS repo, t.file_path AS file_path, "
                    "toFloat(r.confidence) AS confidence",
                    {"start_id": eid},
                )
                for row in rows:
                    conf = float(row.get("confidence") or 0.0)
                    if conf < min_confidence:
                        continue
                    t_id = str(row["id"])
                    if t_id in seen_ids:
                        continue
                    seen_ids.add(t_id)
                    entity = GraphEntity(
                        id=t_id,
                        kind=str(row.get("kind") or ""),
                        name=str(row.get("name") or ""),
                        repo=str(row.get("repo") or ""),
                        file_path=str(row.get("file_path") or ""),
                    )
                    if hop == 1:
                        directly.append(entity)
                    else:
                        transitively.append(entity)
                    affected_repos.add(entity.repo)
                    next_frontier.append(t_id)
            frontier = next_frontier

        total = len(directly) + len(transitively)
        if total > 20:
            risk = ChangeRisk.CRITICAL
        elif total > 10:
            risk = ChangeRisk.HIGH
        elif total > 3:
            risk = ChangeRisk.MEDIUM
        else:
            risk = ChangeRisk.LOW

        return ImpactReport(
            directly_affected=directly,
            transitively_affected=transitively,
            affected_repos=list(affected_repos),
            cross_repo_edges=[],
            risk_level=risk,
        )

    # ── Mutations (create bitemporal version records) ─────────────────────────

    async def apply_mutations(
        self,
        mutations: Sequence[TemporalMutation],
        change_source: ChangeSource,
        commit_sha: str | None = None,
        agent_id: str | None = None,
    ) -> TemporalMutationResult:
        """Apply a batch of mutations, each creating a new temporal record.

        All mutations in the batch share the same ``observed_at`` timestamp.
        - ``create_entity``: new node with ``valid_from=now``, ``valid_until=null``
        - ``update_entity``: close current (``valid_until=now``), create new version
        - ``delete_entity``: close current (``valid_until=now``), no new version
        - ``create_edge``, ``update_edge``, ``delete_edge``: same logic for edges

        See: realtime-temporal-spec.py §A, lines 827–843.
        """
        now = datetime.now(tz=UTC)
        observed_at = now.isoformat()

        entities_created = 0
        entities_updated = 0
        entities_deleted = 0
        edges_created = 0
        edges_deleted = 0
        errors: list[str] = []

        for mutation in mutations:
            try:
                op = mutation.operation

                if op == "create_entity":
                    await self._create_entity(
                        mutation.payload, change_source, commit_sha, agent_id, observed_at
                    )
                    entities_created += 1

                elif op == "update_entity":
                    eid = (
                        mutation.entity_id
                        if mutation.entity_id is not None
                        else str(mutation.payload.get("id", ""))
                    )
                    new_vid = str(mutation.payload.get("version_id") or uuid.uuid4())
                    # Step 1: close the current version
                    await self._client.execute_write(
                        "MATCH (e:Entity {id: $id}) "
                        "WHERE e.valid_until IS NULL "
                        "SET e.valid_until = $valid_until, e.superseded_by = $new_vid",
                        {"id": eid, "valid_until": observed_at, "new_vid": new_vid},
                    )
                    # Step 2: create the new version
                    payload = dict(mutation.payload)
                    payload.setdefault("id", eid)
                    payload["version_id"] = new_vid
                    await self._create_entity(
                        payload, change_source, commit_sha, agent_id, observed_at
                    )
                    entities_updated += 1

                elif op == "delete_entity":
                    eid = (
                        mutation.entity_id
                        if mutation.entity_id is not None
                        else str(mutation.payload.get("id", ""))
                    )
                    await self._client.execute_write(
                        "MATCH (e:Entity {id: $id}) "
                        "WHERE e.valid_until IS NULL "
                        "SET e.valid_until = $valid_until",
                        {"id": eid, "valid_until": observed_at},
                    )
                    entities_deleted += 1

                elif op == "create_edge":
                    await self._create_edge(mutation.payload, change_source, observed_at)
                    edges_created += 1

                elif op == "update_edge":
                    edge_id = (
                        mutation.edge_id
                        if mutation.edge_id is not None
                        else str(mutation.payload.get("edge_id", ""))
                    )
                    new_vid = str(mutation.payload.get("version_id") or uuid.uuid4())
                    # Close old edge version
                    await self._client.execute_write(
                        "MATCH ()-[r:RELATES {edge_id: $edge_id}]->() "
                        "WHERE r.valid_until IS NULL "
                        "SET r.valid_until = $valid_until, r.superseded_by = $new_vid",
                        {"edge_id": edge_id, "valid_until": observed_at, "new_vid": new_vid},
                    )
                    # Create new edge version
                    payload = dict(mutation.payload)
                    payload["edge_id"] = edge_id
                    payload["version_id"] = new_vid
                    await self._create_edge(payload, change_source, observed_at)
                    edges_deleted += 1
                    edges_created += 1

                elif op == "delete_edge":
                    edge_id = (
                        mutation.edge_id
                        if mutation.edge_id is not None
                        else str(mutation.payload.get("edge_id", ""))
                    )
                    await self._client.execute_write(
                        "MATCH ()-[r:RELATES {edge_id: $edge_id}]->() "
                        "WHERE r.valid_until IS NULL "
                        "SET r.valid_until = $valid_until",
                        {"edge_id": edge_id, "valid_until": observed_at},
                    )
                    edges_deleted += 1

            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"{mutation.operation}("
                    f"{mutation.entity_id or mutation.edge_id or '?'}): {exc}"
                )

        return TemporalMutationResult(
            entities_created=entities_created,
            entities_updated=entities_updated,
            entities_deleted=entities_deleted,
            edges_created=edges_created,
            edges_deleted=edges_deleted,
            errors=errors,
            timestamp=now,
        )

    # ── Time-travel queries — TASK-1.4.3 stubs ───────────────────────────────

    async def query_at(
        self,
        cypher: str,
        at_time: datetime,
        params: dict[str, Any] | None = None,
    ) -> Sequence[dict[str, Any]]:
        """Time-travel query — implemented in TASK-1.4.3."""
        raise NotImplementedError("query_at is implemented in TASK-1.4.3")

    async def get_entity_at(
        self,
        entity_id: str,
        at_time: datetime,
    ) -> TemporalGraphEntity | None:
        """Time-travel entity lookup — implemented in TASK-1.4.3."""
        raise NotImplementedError("get_entity_at is implemented in TASK-1.4.3")

    async def get_entity_history(
        self,
        entity_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> Sequence[TemporalGraphEntity]:
        """Entity version history — implemented in TASK-1.4.3."""
        raise NotImplementedError("get_entity_history is implemented in TASK-1.4.3")

    # ── Change-awareness queries — TASK-1.4.3 stubs ──────────────────────────

    async def get_changes_since(
        self,
        since: datetime,
        repo: str | None = None,
        entity_kinds: Sequence[str] | None = None,
    ) -> ChangeSet:
        """Change detection since a timestamp — implemented in TASK-1.4.3."""
        raise NotImplementedError("get_changes_since is implemented in TASK-1.4.3")

    async def get_changes_between(
        self,
        start: datetime,
        end: datetime,
        repo: str | None = None,
    ) -> ChangeSet:
        """Bounded change detection — implemented in TASK-1.4.3."""
        raise NotImplementedError("get_changes_between is implemented in TASK-1.4.3")

    async def get_changes_by_commit(
        self,
        commit_sha: str,
    ) -> ChangeSet:
        """Commit-scoped change detection — implemented in TASK-1.4.3."""
        raise NotImplementedError("get_changes_by_commit is implemented in TASK-1.4.3")

    async def diff_graph(
        self,
        time_a: datetime,
        time_b: datetime,
        repo: str | None = None,
    ) -> GraphDiff:
        """Structural graph diff — implemented in TASK-1.4.3."""
        raise NotImplementedError("diff_graph is implemented in TASK-1.4.3")

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _create_entity(
        self,
        payload: dict[str, Any],
        change_source: ChangeSource,
        commit_sha: str | None,
        agent_id: str | None,
        observed_at: str,
    ) -> None:
        """Insert a new entity version node into FalkorDB."""
        params: dict[str, Any] = {
            "id": str(payload.get("id", "")),
            "version_id": str(payload.get("version_id") or uuid.uuid4()),
            "kind": str(payload.get("kind", "")),
            "name": str(payload.get("name", "")),
            "qualified_name": str(payload.get("qualified_name", "")),
            "repo": str(payload.get("repo", "")),
            "file_path": str(payload.get("file_path", "")),
            "valid_from": observed_at,
            "observed_at": observed_at,
            "change_source": str(change_source),
            "commit_sha": commit_sha,
            "agent_id": agent_id,
        }
        await self._client.execute_write(
            "CREATE (e:Entity {"
            "id: $id, version_id: $version_id, kind: $kind, name: $name, "
            "qualified_name: $qualified_name, repo: $repo, file_path: $file_path, "
            "valid_from: $valid_from, valid_until: null, observed_at: $observed_at, "
            "change_source: $change_source, commit_sha: $commit_sha, "
            "agent_id: $agent_id, superseded_by: null"
            "})",
            params,
        )

    async def _create_edge(
        self,
        payload: dict[str, Any],
        change_source: ChangeSource,
        observed_at: str,
    ) -> None:
        """Insert a new RELATES edge version into FalkorDB."""
        params: dict[str, Any] = {
            "src_id": str(payload.get("source_entity_id", "")),
            "tgt_id": str(payload.get("target_entity_id", "")),
            "edge_id": str(payload.get("edge_id") or uuid.uuid4()),
            "version_id": str(payload.get("version_id") or uuid.uuid4()),
            "relationship": str(payload.get("relationship", "RELATES")),
            "confidence": float(payload.get("confidence", 1.0)),
            "resolution": str(payload.get("resolution", "deterministic")),
            "valid_from": observed_at,
            "observed_at": observed_at,
            "change_source": str(change_source),
        }
        await self._client.execute_write(
            "MATCH (src:Entity {id: $src_id}), (tgt:Entity {id: $tgt_id}) "
            "WHERE src.valid_until IS NULL AND tgt.valid_until IS NULL "
            "CREATE (src)-[r:RELATES {"
            "edge_id: $edge_id, version_id: $version_id, "
            "relationship: $relationship, confidence: $confidence, "
            "resolution: $resolution, valid_from: $valid_from, "
            "valid_until: null, observed_at: $observed_at, "
            "change_source: $change_source"
            "}]->(tgt)",
            params,
        )
