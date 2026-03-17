"""Temporal graph value types for the Tractable framework.

All graph nodes and edges carry bitemporal metadata (valid_from/valid_until)
so nothing is ever deleted — only versioned.

Sources:
- realtime-temporal-spec.py §A — Temporal Graph Model
- realtime-temporal-spec.py §C — ChangeNotification, ChangeRelevance
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from tractable.types.enums import ChangeRelevance, ChangeSource


class TemporalMetadata(BaseModel):
    """
    Bitemporal metadata carried by every graph node and edge.

    - valid_from / valid_until: when this fact was TRUE in the codebase
    - observed_at: when the system first detected this fact
    - superseded_by: pointer to the new version, if this one was replaced

    Current state: ``valid_until`` is ``None``.
    Historical: ``valid_until`` is set to when the entity changed/was removed.
    """

    valid_from: datetime
    valid_until: datetime | None = None
    observed_at: datetime
    superseded_by: str | None = None
    change_source: ChangeSource
    commit_sha: str | None = None
    agent_id: str | None = None


class TemporalGraphEntity(BaseModel):
    """A graph node with full temporal tracking."""

    id: str
    version_id: str
    kind: str
    name: str
    qualified_name: str
    repo: str
    file_path: str
    line_start: int | None = None
    line_end: int | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    temporal: TemporalMetadata

    @property
    def is_current(self) -> bool:
        return self.temporal.valid_until is None


class TemporalEdge(BaseModel):
    """A graph edge with full temporal tracking."""

    edge_id: str
    version_id: str
    source_entity_id: str
    target_entity_id: str
    relationship: str
    confidence: float
    resolution: str  # EdgeConfidence value
    properties: dict[str, Any] = Field(default_factory=dict)
    temporal: TemporalMetadata

    @property
    def is_current(self) -> bool:
        return self.temporal.valid_until is None


class EntityModification(BaseModel):
    """Tracks what specifically changed in an entity."""

    entity_id: str
    previous_version: TemporalGraphEntity
    current_version: TemporalGraphEntity
    changed_fields: list[str] = Field(default_factory=list)
    change_description: str


class ChangeSet(BaseModel):
    """A set of changes that occurred in a time range."""

    time_range_start: datetime
    time_range_end: datetime
    repo_filter: str | None = None

    entities_added: list[TemporalGraphEntity] = []
    entities_modified: list[EntityModification] = []
    entities_removed: list[TemporalGraphEntity] = []

    edges_added: list[TemporalEdge] = []
    edges_removed: list[TemporalEdge] = []

    commits: list[str] = Field(default_factory=list)
    agents_involved: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return (
            len(self.entities_added) == 0
            and len(self.entities_modified) == 0
            and len(self.entities_removed) == 0
            and len(self.edges_added) == 0
            and len(self.edges_removed) == 0
        )

    @property
    def summary(self) -> str:
        """Human-readable summary for agent context injection."""
        parts: list[str] = []
        if self.entities_added:
            parts.append(f"{len(self.entities_added)} entities added")
        if self.entities_modified:
            parts.append(f"{len(self.entities_modified)} entities modified")
        if self.entities_removed:
            parts.append(f"{len(self.entities_removed)} entities removed")
        if self.edges_added:
            parts.append(f"{len(self.edges_added)} new relationships")
        if self.edges_removed:
            parts.append(f"{len(self.edges_removed)} relationships removed")
        return "; ".join(parts) if parts else "No changes"


class GraphDiff(BaseModel):
    """Structural diff between two graph states."""

    time_a: datetime
    time_b: datetime
    added_entities: list[TemporalGraphEntity] = []
    removed_entities: list[TemporalGraphEntity] = []
    modified_entities: list[EntityModification] = []
    added_edges: list[TemporalEdge] = []
    removed_edges: list[TemporalEdge] = []
    repos_affected: list[str] = Field(default_factory=list)

    def for_repo(self, repo: str) -> GraphDiff:
        """Return a copy of this diff filtered to changes in ``repo``."""
        return GraphDiff(
            time_a=self.time_a,
            time_b=self.time_b,
            added_entities=[e for e in self.added_entities if e.repo == repo],
            removed_entities=[e for e in self.removed_entities if e.repo == repo],
            modified_entities=[
                m for m in self.modified_entities if m.current_version.repo == repo
            ],
            added_edges=[
                e
                for e in self.added_edges
                if repo in (e.source_entity_id, e.target_entity_id)
            ],
            removed_edges=[
                e
                for e in self.removed_edges
                if repo in (e.source_entity_id, e.target_entity_id)
            ],
            repos_affected=[repo],
        )


class TemporalMutation(BaseModel):
    """A single graph mutation that creates temporal records."""

    operation: Literal[
        "create_entity",
        "update_entity",
        "delete_entity",
        "create_edge",
        "update_edge",
        "delete_edge",
    ]
    entity_id: str | None = None
    edge_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class TemporalMutationResult(BaseModel):
    """Result of applying a batch of temporal mutations."""

    entities_created: int
    entities_updated: int
    entities_deleted: int
    edges_created: int
    edges_deleted: int
    errors: list[str] = Field(default_factory=list)
    timestamp: datetime


class ChangeNotification(BaseModel):
    """Notification sent to an agent about a change that may affect their domain."""

    target_agent_id: str
    repo_name: str
    relevance: ChangeRelevance
    change_summary: str
    affected_entity_ids: list[str] = Field(default_factory=list)
    cross_repo_edges_affected: list[str] = Field(default_factory=list)
    commit_sha: str
    requires_action: bool
