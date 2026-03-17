"""CodeGraph and TemporalCodeGraph Protocols.

Sources:
- tech-spec.py §2.2 — CodeGraph Protocol
- realtime-temporal-spec.py §B — TemporalCodeGraph Protocol
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from tractable.types.enums import ChangeSource
from tractable.types.graph import (
    CrossRepoEdge,
    GraphEntity,
    GraphMutation,
    ImpactReport,
    MutationResult,
    RepoGraphSummary,
    Subgraph,
)
from tractable.types.temporal import (
    ChangeSet,
    GraphDiff,
    TemporalGraphEntity,
    TemporalMutation,
    TemporalMutationResult,
)


@runtime_checkable
class CodeGraph(Protocol):
    """
    Interface to the unified code knowledge graph spanning all registered
    repositories. Agents use this for scoped context retrieval, blast-radius
    analysis, and cross-repo awareness.
    """

    async def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> Sequence[dict[str, Any]]:
        """Execute a read-only Cypher query."""
        ...

    async def get_entity(self, entity_id: str) -> GraphEntity | None:
        """Retrieve a single node by ID."""
        ...

    async def get_neighborhood(
        self,
        entity_id: str,
        depth: int = 2,
        min_confidence: float = 0.7,
    ) -> Subgraph:
        """Get the subgraph around an entity within N hops."""
        ...

    async def impact_analysis(
        self,
        entity_ids: Sequence[str],
        depth: int = 3,
        min_confidence: float = 0.5,
    ) -> ImpactReport:
        """Compute blast radius for proposed changes to given entities."""
        ...

    async def get_repo_boundary_edges(
        self,
        repo_name: str,
    ) -> Sequence[CrossRepoEdge]:
        """Get all edges that cross this repo's boundary."""
        ...

    async def get_repo_summary(self, repo_name: str) -> RepoGraphSummary:
        """Compact summary of a repo's entities for agent context."""
        ...

    async def mutate(self, mutations: Sequence[GraphMutation]) -> MutationResult:
        """Apply graph updates (add/remove/update nodes and edges)."""
        ...


@runtime_checkable
class TemporalCodeGraph(Protocol):
    """
    The code graph with full temporal awareness.

    Current-state queries (valid_until IS NULL) are the fast path used by
    agents during normal operation. Temporal queries enable time-travel and
    change-detection on wake-up.
    """

    # ── Current-state queries (fast path) ─────────────────────────────

    async def query_current(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> Sequence[dict[str, Any]]:
        """
        Query only current (live) entities.
        Implicitly filters to valid_until IS NULL.
        """
        ...

    async def get_current_entity(
        self, entity_id: str
    ) -> TemporalGraphEntity | None:
        """Get the current version of an entity."""
        ...

    async def impact_analysis_current(
        self,
        entity_ids: Sequence[str],
        depth: int = 3,
        min_confidence: float = 0.5,
    ) -> ImpactReport:
        """Blast radius against current graph state."""
        ...

    # ── Time-travel queries ────────────────────────────────────────────

    async def query_at(
        self,
        cypher: str,
        at_time: datetime,
        params: dict[str, Any] | None = None,
    ) -> Sequence[dict[str, Any]]:
        """
        Query the graph as it existed at a specific point in time.
        Filters to valid_from <= at_time < valid_until.
        """
        ...

    async def get_entity_at(
        self,
        entity_id: str,
        at_time: datetime,
    ) -> TemporalGraphEntity | None:
        """Get the version of an entity that was current at a given time."""
        ...

    async def get_entity_history(
        self,
        entity_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> Sequence[TemporalGraphEntity]:
        """
        Get all versions of an entity over time.
        Returns ordered by valid_from ascending.
        """
        ...

    # ── Change-awareness queries ───────────────────────────────────────

    async def get_changes_since(
        self,
        since: datetime,
        repo: str | None = None,
        entity_kinds: Sequence[str] | None = None,
    ) -> ChangeSet:
        """
        What changed in the graph since a given time?
        Primary query agents use on wake-up.
        """
        ...

    async def get_changes_between(
        self,
        start: datetime,
        end: datetime,
        repo: str | None = None,
    ) -> ChangeSet:
        """What changed between two points in time?"""
        ...

    async def get_changes_by_commit(
        self,
        commit_sha: str,
    ) -> ChangeSet:
        """What graph mutations resulted from a specific commit?"""
        ...

    async def diff_graph(
        self,
        time_a: datetime,
        time_b: datetime,
        repo: str | None = None,
    ) -> GraphDiff:
        """Structural diff between two points in time."""
        ...

    # ── Mutation (creates temporal records) ───────────────────────────

    async def apply_mutations(
        self,
        mutations: Sequence[TemporalMutation],
        change_source: ChangeSource,
        commit_sha: str | None = None,
        agent_id: str | None = None,
    ) -> TemporalMutationResult:
        """
        Apply mutations that create new temporal versions.

        For entity updates: old version gets valid_until set, new version
        created with valid_from = now. For deletions: valid_until set, no
        new version created.
        """
        ...
