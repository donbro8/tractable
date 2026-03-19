"""FalkorDB temporal schema constants and Cypher query templates.

Sources:
- realtime-temporal-spec.py §A — TemporalMetadata fields (lines 49-69)
- realtime-temporal-spec.py §H — Index strategy (lines 922-930)
"""

from __future__ import annotations

# ── Property lists ────────────────────────────────────────────────────────────

# Every Entity node must carry these properties.
ENTITY_PROPERTIES: list[str] = [
    "id",
    "version_id",
    "kind",
    "name",
    "qualified_name",
    "repo",
    "file_path",
    "valid_from",
    "valid_until",
    "observed_at",
    "change_source",
    "commit_sha",
    "agent_id",
    "superseded_by",
]

# Every RELATES edge must carry these properties.
EDGE_PROPERTIES: list[str] = [
    "edge_id",
    "version_id",
    "relationship",
    "confidence",
    "resolution",
    "valid_from",
    "valid_until",
    "observed_at",
    "change_source",
]

# ── Cypher query templates ────────────────────────────────────────────────────

# Retrieve all current (live) entities (valid_until IS NULL).
# Uses the entity_current index.
QUERY_CURRENT_ENTITIES = (
    "MATCH (e:Entity) "
    "WHERE e.valid_until IS NULL "
    "RETURN e"
)

# Retrieve all current entities for a specific repo.
QUERY_CURRENT_ENTITIES_BY_REPO = (
    "MATCH (e:Entity) "
    "WHERE e.valid_until IS NULL AND e.repo = $repo "
    "RETURN e"
)

# Retrieve the current version of a single entity by its stable id.
QUERY_CURRENT_ENTITY_BY_ID = (
    "MATCH (e:Entity {id: $id}) "
    "WHERE e.valid_until IS NULL "
    "RETURN e"
)

# Create a new entity version (valid_until is null = current).
UPSERT_ENTITY = (
    "CREATE (e:Entity {"
    "id: $id, version_id: $version_id, kind: $kind, name: $name, "
    "qualified_name: $qualified_name, repo: $repo, file_path: $file_path, "
    "valid_from: $valid_from, valid_until: null, observed_at: $observed_at, "
    "change_source: $change_source, commit_sha: $commit_sha, "
    "agent_id: $agent_id, superseded_by: null"
    "})"
)

# Close an entity version: record valid_until and which version supersedes it.
CLOSE_ENTITY_VERSION = (
    "MATCH (e:Entity {id: $id, version_id: $version_id}) "
    "SET e.valid_until = $valid_until, e.superseded_by = $superseded_by"
)

# All versions of an entity ordered chronologically.
QUERY_ENTITY_HISTORY = (
    "MATCH (e:Entity {id: $id}) "
    "RETURN e "
    "ORDER BY e.valid_from ASC"
)

# Entities observed after a given timestamp — uses the entity_observed index.
# Critical perf path: called on every agent wake-up via get_changes_since().
# See: realtime-temporal-spec.py §H — (observed_at) index for get_changes_since.
QUERY_ENTITIES_SINCE = (
    "MATCH (e:Entity) "
    "WHERE e.observed_at >= $since "
    "RETURN e "
    "ORDER BY e.observed_at ASC"
)

# Same as QUERY_ENTITIES_SINCE but scoped to a single repo.
QUERY_ENTITIES_SINCE_BY_REPO = (
    "MATCH (e:Entity) "
    "WHERE e.observed_at >= $since AND e.repo = $repo "
    "RETURN e "
    "ORDER BY e.observed_at ASC"
)
