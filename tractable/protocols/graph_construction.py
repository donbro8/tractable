"""CodeParser and FuzzyResolver Protocols plus their supporting value types.

Source: tech-spec.py §2.4
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from tractable.types.enums import EdgeConfidence
from tractable.types.graph import GraphEntity

# ── Supporting value types ─────────────────────────────────────────────


class ParsedEntity(BaseModel):
    """A code entity extracted by a parser."""

    kind: str  # "function" | "class" | "import" | etc.
    name: str
    qualified_name: str
    file_path: str
    line_start: int
    line_end: int
    properties: dict[str, Any] = Field(default_factory=dict)


class ParsedRelationship(BaseModel):
    """A relationship between two parsed entities."""

    source_qualified_name: str
    target_qualified_name: str
    relationship: str
    confidence: float = 1.0
    resolution: EdgeConfidence = EdgeConfidence.DETERMINISTIC


class UnresolvedReference(BaseModel):
    """A string reference that static parsing could not resolve."""

    source_file: str
    source_line: int
    reference_string: str
    context_snippet: str
    likely_kind: str | None = None


class ParseResult(BaseModel):
    """Output of parsing a single source file."""

    file_path: str
    language: str
    entities: list[ParsedEntity] = []
    relationships: list[ParsedRelationship] = []
    unresolved_references: list[UnresolvedReference] = []


class ResolvedReference(BaseModel):
    """An unresolved reference that has been resolved (or declared unresolvable)."""

    original: UnresolvedReference
    target_entity_id: str | None  # None if unresolvable
    confidence: float
    resolution: EdgeConfidence
    reasoning: str


# ── Protocols ──────────────────────────────────────────────────────────


@runtime_checkable
class CodeParser(Protocol):
    """
    Parses source files into structured entities.
    One parser per language or file type. Composed into pipelines.
    """

    @property
    def supported_extensions(self) -> frozenset[str]:
        """File extensions this parser handles, e.g. ``{'.py', '.pyi'}``."""
        ...

    async def parse_file(
        self,
        file_path: str,
        content: bytes,
    ) -> ParseResult:
        """Extract entities and relationships from a single file."""
        ...


@runtime_checkable
class FuzzyResolver(Protocol):
    """Resolves references that static parsing could not handle."""

    async def resolve_batch(
        self,
        references: Sequence[UnresolvedReference],
        candidate_entities: Sequence[GraphEntity],
    ) -> Sequence[ResolvedReference]:
        """Attempt to resolve a batch of unresolved references."""
        ...
