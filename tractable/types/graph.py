"""Code graph value types for the Tractable framework.

All models are pure data containers — no implementation logic.

Source: tech-spec.py §2.2 — Code Graph Protocol value types.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from tractable.types.enums import ChangeRisk, EdgeConfidence


class GraphEntity(BaseModel):
    """A node in the code knowledge graph."""

    id: str
    kind: str  # "function" | "class" | "config_key" | "infra_resource" | etc.
    name: str
    repo: str
    file_path: str
    line: int | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class CrossRepoEdge(BaseModel):
    """A directed edge connecting entities across repositories."""

    source_entity_id: str
    source_repo: str
    target_entity_id: str
    target_repo: str
    relationship: str  # "CALLS" | "REFERENCES" | "CONFIGURES" | etc.
    confidence: float
    resolution: EdgeConfidence


class Subgraph(BaseModel):
    """A subset of the full code knowledge graph."""

    nodes: list[GraphEntity] = []
    edges: list[CrossRepoEdge] = []


class ImpactReport(BaseModel):
    """Blast-radius analysis for a proposed change."""

    directly_affected: list[GraphEntity] = []
    transitively_affected: list[GraphEntity] = []
    affected_repos: list[str] = Field(default_factory=list)
    cross_repo_edges: list[CrossRepoEdge] = []
    risk_level: ChangeRisk


class RepoGraphSummary(BaseModel):
    """Compact representation for agent context injection."""

    repo_name: str
    total_entities: int
    key_modules: list[str] = Field(default_factory=list)
    public_interfaces: list[GraphEntity] = []
    cross_repo_dependencies: list[CrossRepoEdge] = []
    summary_text: str  # LLM-generated architectural summary


class GraphMutation(BaseModel):
    """A single graph write operation."""

    operation: Literal[
        "create_node",
        "update_node",
        "delete_node",
        "create_edge",
        "update_edge",
        "delete_edge",
    ]
    payload: dict[str, Any] = Field(default_factory=dict)


class MutationResult(BaseModel):
    """Result of applying a batch of graph mutations."""

    applied: int
    errors: list[str] = Field(default_factory=list)
