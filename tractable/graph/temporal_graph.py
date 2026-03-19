"""FalkorDB implementation of the TemporalCodeGraph protocol.

Implements current-state operations (TASK-1.4.2) and temporal/change-awareness
query methods (TASK-1.4.3).

Sources:
- realtime-temporal-spec.py §B — TemporalCodeGraph Protocol (lines 144–264)
- realtime-temporal-spec.py §A — temporal mutation patterns (lines 827–843)
- realtime-temporal-spec.py §F — agent catchup query example (lines 846–860)
- realtime-temporal-spec.py §H — index strategy, PHASE 1 additions (lines 927–930)
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
    EntityModification,
    GraphDiff,
    TemporalGraphEntity,
    TemporalMetadata,
    TemporalMutation,
    TemporalMutationResult,
)

# ── Shared Cypher return clause for entity properties ─────────────────────────

_ENTITY_RETURN = (
    "e.id AS id, e.version_id AS version_id, e.kind AS kind, "
    "e.name AS name, e.qualified_name AS qualified_name, "
    "e.repo AS repo, e.file_path AS file_path, "
    "e.valid_from AS valid_from, e.valid_until AS valid_until, "
    "e.observed_at AS observed_at, e.change_source AS change_source, "
    "e.commit_sha AS commit_sha, e.agent_id AS agent_id, "
    "e.superseded_by AS superseded_by"
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


def _inject_at_filter(cypher: str, at_str: str) -> str:  # noqa: ARG001
    """Inject a point-in-time validity filter into a Cypher query.

    Replaces ``WHERE e.valid_until IS NULL`` (if present) or inserts before
    RETURN.  The ``__at`` parameter must be passed when executing the query.
    """
    temporal_cond = "e.valid_from <= $__at AND (e.valid_until IS NULL OR e.valid_until > $__at)"
    upper = cypher.upper()

    # Replace an existing current-state filter if present
    current_filter = "E.VALID_UNTIL IS NULL"
    if current_filter in upper:
        idx = upper.index(current_filter)
        return cypher[:idx] + temporal_cond + cypher[idx + len(current_filter):]

    where_pos = upper.find(" WHERE ")
    return_pos = upper.find(" RETURN ")

    if where_pos != -1 and (return_pos == -1 or where_pos < return_pos):
        insert = where_pos + len(" WHERE ")
        return cypher[:insert] + temporal_cond + " AND " + cypher[insert:]
    if return_pos != -1:
        return cypher[:return_pos] + f" WHERE {temporal_cond}" + cypher[return_pos:]
    return cypher + f" WHERE {temporal_cond}"


def _compute_changed_fields(
    prev: TemporalGraphEntity, curr: TemporalGraphEntity
) -> list[str]:
    """Return a list of field names that differ between two entity versions."""
    scalar_fields = ["kind", "name", "qualified_name", "repo", "file_path",
                     "line_start", "line_end"]
    return [f for f in scalar_fields if getattr(prev, f) != getattr(curr, f)]


# ── FalkorDBTemporalGraph ─────────────────────────────────────────────────────


class FalkorDBTemporalGraph:
    """FalkorDB-backed implementation of the TemporalCodeGraph protocol.

    Current-state queries (``valid_until IS NULL``) are the fast path used by
    agents during normal operation. Temporal/change-awareness methods use the
    ``(observed_at)`` and ``(id, valid_from)`` indexes for efficient time-travel
    and change detection (realtime-temporal-spec.py §H).
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
            f"MATCH (e:Entity {{id: $id}}) WHERE e.valid_until IS NULL RETURN {_ENTITY_RETURN}",
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

    # ── Time-travel queries (TASK-1.4.3) ─────────────────────────────────────

    async def query_at(
        self,
        cypher: str,
        at_time: datetime,
        params: dict[str, Any] | None = None,
    ) -> Sequence[dict[str, Any]]:
        """Execute a Cypher query scoped to graph state at ``at_time``.

        Injects ``WHERE e.valid_from <= $at AND (e.valid_until IS NULL OR
        e.valid_until > $at)`` into the query automatically.  The query must
        use ``e`` as the entity node alias.
        """
        at_str = at_time.isoformat()
        filtered = _inject_at_filter(cypher, at_str)
        merged = dict(params or {})
        merged["__at"] = at_str
        return await self._client.execute(filtered, merged)

    async def get_entity_at(
        self,
        entity_id: str,
        at_time: datetime,
    ) -> TemporalGraphEntity | None:
        """Return the entity version that was current at ``at_time``, or ``None``."""
        at_str = at_time.isoformat()
        rows = await self._client.execute(
            f"MATCH (e:Entity {{id: $id}}) "
            f"WHERE e.valid_from <= $at AND (e.valid_until IS NULL OR e.valid_until > $at) "
            f"RETURN {_ENTITY_RETURN}",
            {"id": entity_id, "at": at_str},
        )
        if not rows:
            return None
        return _row_to_entity(rows[0])

    async def get_entity_history(
        self,
        entity_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> Sequence[TemporalGraphEntity]:
        """Return all versions of an entity ordered by ``valid_from ASC``.

        Optional ``since`` / ``until`` bound the versions returned by their
        ``valid_from`` timestamp.
        """
        conditions = ["e.id = $id"]
        params: dict[str, Any] = {"id": entity_id}
        if since is not None:
            conditions.append("e.valid_from >= $since")
            params["since"] = since.isoformat()
        if until is not None:
            conditions.append("e.valid_from <= $until")
            params["until"] = until.isoformat()
        where = " AND ".join(conditions)
        rows = await self._client.execute(
            f"MATCH (e:Entity) WHERE {where} RETURN {_ENTITY_RETURN} ORDER BY e.valid_from",
            params,
        )
        return [_row_to_entity(r) for r in rows]

    # ── Change-awareness queries (TASK-1.4.3) ────────────────────────────────

    async def get_changes_since(
        self,
        since: datetime,
        repo: str | None = None,
        entity_kinds: Sequence[str] | None = None,
    ) -> ChangeSet:
        """Return all changes observed since ``since``.

        Uses the ``(observed_at)`` index — critical performance path for agent
        wake-up catchup (realtime-temporal-spec.py §H).  Classification:
        - ``entities_added``    — new entity, no prior version before ``since``
        - ``entities_modified`` — entity updated; prior version existed
        - ``entities_removed``  — entity deleted; no current version remains
        """
        since_str = since.isoformat()
        now = datetime.now(tz=UTC)
        return await self._collect_changes(since_str, None, now.isoformat(), repo, entity_kinds)

    async def get_changes_between(
        self,
        start: datetime,
        end: datetime,
        repo: str | None = None,
    ) -> ChangeSet:
        """Return changes where ``start <= observed_at < end``."""
        return await self._collect_changes(
            start.isoformat(), start.isoformat(), end.isoformat(), repo, None
        )

    async def get_changes_by_commit(
        self,
        commit_sha: str,
    ) -> ChangeSet:
        """Return all changes attributed to ``commit_sha``."""
        rows = await self._client.execute(
            f"MATCH (e:Entity) WHERE e.commit_sha = $sha RETURN {_ENTITY_RETURN}",
            {"sha": commit_sha},
        )
        now = datetime.now(tz=UTC)
        entities_added: list[TemporalGraphEntity] = []
        entities_modified: list[EntityModification] = []
        entities_removed: list[TemporalGraphEntity] = []
        commits: set[str] = set()

        for row in rows:
            entity = _row_to_entity(row)
            commits.add(commit_sha)
            valid_until_raw: Any = row.get("valid_until")
            if valid_until_raw is not None:
                entities_removed.append(entity)
            elif row.get("superseded_by") is None:
                entities_added.append(entity)
            else:
                entities_added.append(entity)

        earliest = min((e.temporal.observed_at for e in entities_added), default=now)
        return ChangeSet(
            time_range_start=earliest,
            time_range_end=now,
            entities_added=entities_added,
            entities_modified=entities_modified,
            entities_removed=entities_removed,
            commits=list(commits),
        )

    async def diff_graph(
        self,
        time_a: datetime,
        time_b: datetime,
        repo: str | None = None,
    ) -> GraphDiff:
        """Compute a structural diff between graph state at ``time_a`` and ``time_b``."""
        repo_filter = " AND e.repo = $repo" if repo else ""
        params_a: dict[str, Any] = {"at": time_a.isoformat()}
        params_b: dict[str, Any] = {"at": time_b.isoformat()}
        if repo:
            params_a["repo"] = repo
            params_b["repo"] = repo

        at_clause = (
            "e.valid_from <= $at AND (e.valid_until IS NULL OR e.valid_until > $at)"
        )
        q = f"MATCH (e:Entity) WHERE {at_clause}{repo_filter} RETURN {_ENTITY_RETURN}"

        rows_a = await self._client.execute(q, params_a)
        rows_b = await self._client.execute(q, params_b)

        state_a: dict[str, TemporalGraphEntity] = {
            str(r["id"]): _row_to_entity(r) for r in rows_a
        }
        state_b: dict[str, TemporalGraphEntity] = {
            str(r["id"]): _row_to_entity(r) for r in rows_b
        }

        ids_a = set(state_a)
        ids_b = set(state_b)

        added_entities = [state_b[eid] for eid in ids_b - ids_a]
        removed_entities = [state_a[eid] for eid in ids_a - ids_b]
        modified_entities: list[EntityModification] = []
        for eid in ids_a & ids_b:
            prev = state_a[eid]
            curr = state_b[eid]
            if prev.version_id != curr.version_id:
                modified_entities.append(
                    EntityModification(
                        entity_id=eid,
                        previous_version=prev,
                        current_version=curr,
                        changed_fields=_compute_changed_fields(prev, curr),
                        change_description=f"Entity {eid} changed between snapshots",
                    )
                )

        repos_affected: list[str] = list(
            {e.repo for e in added_entities}
            | {e.repo for e in removed_entities}
            | {m.current_version.repo for m in modified_entities}
        )

        return GraphDiff(
            time_a=time_a,
            time_b=time_b,
            added_entities=added_entities,
            removed_entities=removed_entities,
            modified_entities=modified_entities,
            repos_affected=repos_affected,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _collect_changes(
        self,
        since_str: str,
        obs_start: str | None,
        obs_end: str,
        repo: str | None,
        entity_kinds: Sequence[str] | None,
    ) -> ChangeSet:
        """Core logic shared by get_changes_since and get_changes_between.

        Three-query approach:
        1. Current entities observed in window  → added or modified candidates
        2. Prior versions for modified detection (JOIN-style Cypher, no N+1)
        3. Closed entities in window            → removed candidates
        """
        # ── Build filter clauses ───────────────────────────────────────────
        repo_filter = " AND e.repo = $repo" if repo else ""
        params: dict[str, Any] = {"since": since_str, "obs_end": obs_end}
        if repo:
            params["repo"] = repo

        kind_clause = ""
        if entity_kinds:
            inlined = ", ".join(f"'{k}'" for k in entity_kinds)
            kind_clause = f" AND e.kind IN [{inlined}]"

        obs_start_clause = ""
        if obs_start:
            obs_start_clause = " AND e.observed_at >= $obs_start"
            params["obs_start"] = obs_start

        # ── Q1: current entities observed in window (uses observed_at index) ─
        q1 = (
            f"MATCH (e:Entity) "
            f"WHERE e.observed_at <= $obs_end{obs_start_clause} AND e.valid_until IS NULL"
            f"{repo_filter}{kind_clause} "
            f"RETURN {_ENTITY_RETURN}"
        )
        current_rows = await self._client.execute(q1, params)
        current_by_id: dict[str, dict[str, Any]] = {
            str(r["id"]): r for r in current_rows
        }

        # ── Q2: prior versions via nested MATCH — determines added vs modified ─
        prior_by_id: dict[str, TemporalGraphEntity] = {}
        if current_by_id:
            q2_params: dict[str, Any] = {"since": since_str, "obs_end": obs_end}
            if repo:
                q2_params["repo"] = repo
            q2 = (
                f"MATCH (new_e:Entity) "
                f"WHERE new_e.observed_at <= $obs_end{obs_start_clause} "
                f"AND new_e.valid_until IS NULL{repo_filter}{kind_clause} "
                f"MATCH (e:Entity {{id: new_e.id}}) "
                f"WHERE e.valid_from < $since "
                f"RETURN {_ENTITY_RETURN} "
                f"ORDER BY e.id, e.valid_from DESC"
            )
            if obs_start:
                q2_params["obs_start"] = obs_start
            prior_rows = await self._client.execute(q2, q2_params)
            for row in prior_rows:
                eid = str(row["id"])
                if eid not in prior_by_id:
                    prior_by_id[eid] = _row_to_entity(row)

        # ── Q3: closed entities in window — candidates for removed ────────────
        q3 = (
            f"MATCH (e:Entity) "
            f"WHERE e.valid_until >= $since AND e.valid_until <= $obs_end "
            f"AND e.valid_until IS NOT NULL{repo_filter} "
            f"RETURN {_ENTITY_RETURN}"
        )
        q3_params: dict[str, Any] = {"since": since_str, "obs_end": obs_end}
        if repo:
            q3_params["repo"] = repo
        closed_rows = await self._client.execute(q3, q3_params)
        closed_entity_ids: set[str] = {str(r["id"]) for r in closed_rows}

        # ── Classify ───────────────────────────────────────────────────────────
        entities_added: list[TemporalGraphEntity] = []
        entities_modified: list[EntityModification] = []
        commits: set[str] = set()
        agents: set[str] = set()

        for eid, row in current_by_id.items():
            entity = _row_to_entity(row)
            cs = entity.temporal.commit_sha
            if cs:
                commits.add(cs)
            ai = entity.temporal.agent_id
            if ai:
                agents.add(ai)

            prior = prior_by_id.get(eid)
            if prior is not None:
                entities_modified.append(
                    EntityModification(
                        entity_id=eid,
                        previous_version=prior,
                        current_version=entity,
                        changed_fields=_compute_changed_fields(prior, entity),
                        change_description=f"Entity {eid} was modified",
                    )
                )
            else:
                entities_added.append(entity)

        # Removed: closed in window AND no current version in Q1
        entities_removed: list[TemporalGraphEntity] = []
        removed_ids = closed_entity_ids - set(current_by_id)
        for row in closed_rows:
            eid = str(row["id"])
            if eid in removed_ids:
                entities_removed.append(_row_to_entity(row))
                removed_ids.discard(eid)  # deduplicate

        since_dt = datetime.fromisoformat(since_str)
        obs_end_dt = datetime.fromisoformat(obs_end)

        return ChangeSet(
            time_range_start=since_dt,
            time_range_end=obs_end_dt,
            repo_filter=repo,
            entities_added=entities_added,
            entities_modified=entities_modified,
            entities_removed=entities_removed,
            commits=list(commits),
            agents_involved=list(agents),
        )

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
