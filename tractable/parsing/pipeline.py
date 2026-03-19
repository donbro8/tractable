"""GraphConstructionPipeline — orchestrates initial repository ingestion.

Clones a registered repository, parses every supported source file, converts
the extracted entities and relationships into :class:`TemporalMutation` records,
and writes them to the :class:`TemporalCodeGraph` in batches.

Source: tech-spec.py §2.4 (CodeParser section), plan.md Phase 1 Week 2.
"""

from __future__ import annotations

import fnmatch
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

from tractable.parsing.parsers.python_parser import PythonParser
from tractable.protocols.code_graph import TemporalCodeGraph
from tractable.protocols.graph_construction import CodeParser, ParsedEntity, ParsedRelationship
from tractable.providers.factory import create_git_provider
from tractable.types.config import RepositoryRegistration
from tractable.types.enums import ChangeSource
from tractable.types.temporal import TemporalMutation

log = structlog.get_logger(__name__)

# ── IngestResult ──────────────────────────────────────────────────────────────


class IngestResult(BaseModel):
    """Summary of a completed initial ingestion run.

    Internal to the pipeline — not exported from tractable.types.
    """

    files_parsed: int
    entities_created: int
    relationships_created: int
    unresolved_references: int
    duration_seconds: float
    errors: list[str] = Field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

_BATCH_SIZE = 500


def _is_ignored(rel_path: str, patterns: list[str]) -> bool:
    """Return True if *rel_path* matches any glob ignore pattern."""
    for pattern in patterns:
        if fnmatch.fnmatch(rel_path, pattern):
            return True
        # Also check if any path segment matches (e.g. __pycache__/** should
        # match __pycache__/foo.cpython-311.pyc regardless of depth).
        if fnmatch.fnmatch(rel_path.replace("\\", "/"), pattern):
            return True
    return False


def _is_in_scope(rel_path: str, registration: RepositoryRegistration) -> bool:
    """Return True if *rel_path* is within the optional scope restrictions."""
    scope = registration.scope
    if scope is None:
        return True
    ext = Path(rel_path).suffix
    if scope.allowed_extensions and ext not in scope.allowed_extensions:
        return False
    if scope.allowed_paths and not any(
        rel_path.startswith(p.rstrip("/")) for p in scope.allowed_paths
    ):
        return False
    return not scope.deny_paths or not any(
        rel_path.startswith(p.rstrip("/")) for p in scope.deny_paths
    )


def _entity_to_mutation(entity: ParsedEntity, repo: str) -> TemporalMutation:
    """Convert a ParsedEntity to a create_entity TemporalMutation."""
    payload: dict[str, Any] = {
        "id": entity.qualified_name,
        "kind": entity.kind,
        "name": entity.name,
        "qualified_name": entity.qualified_name,
        "repo": repo,
        "file_path": entity.file_path,
        "line_start": entity.line_start,
        "line_end": entity.line_end,
        "properties": entity.properties,
    }
    return TemporalMutation(
        operation="create_entity",
        entity_id=entity.qualified_name,
        payload=payload,
    )


def _relationship_to_mutation(rel: ParsedRelationship, repo: str) -> TemporalMutation:
    """Convert a ParsedRelationship to a create_edge TemporalMutation."""
    payload: dict[str, Any] = {
        "repo": repo,
        "source_entity_id": rel.source_qualified_name,
        "target_entity_id": rel.target_qualified_name,
        "relationship": rel.relationship,
        "confidence": rel.confidence,
        "resolution": rel.resolution,
    }
    return TemporalMutation(
        operation="create_edge",
        edge_id=f"{rel.source_qualified_name}:{rel.relationship}:{rel.target_qualified_name}",
        payload=payload,
    )


def _get_head_sha(local_path: str) -> str | None:
    """Return the HEAD commit SHA of the local git clone, or None on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=local_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# ── Pipeline ──────────────────────────────────────────────────────────────────


class GraphConstructionPipeline:
    """Orchestrates the initial ingestion of a registered repository.

    Parsers are selected by file extension. A :class:`PythonParser` is
    registered by default; additional parsers can be injected via the
    constructor for testing or extensibility.
    """

    def __init__(
        self,
        extra_parsers: list[CodeParser] | None = None,
    ) -> None:
        default_parsers: list[CodeParser] = [PythonParser()]
        self._parsers: list[CodeParser] = default_parsers + (extra_parsers or [])

    def _parser_for(self, file_path: str) -> CodeParser | None:
        ext = Path(file_path).suffix
        for parser in self._parsers:
            if ext in parser.supported_extensions:
                return parser
        return None

    async def ingest_repository(
        self,
        registration: RepositoryRegistration,
        graph: TemporalCodeGraph,
    ) -> IngestResult:
        """Clone *registration.git_url*, parse all supported files, and write to *graph*.

        Files matching ``registration.ignore_patterns`` or outside
        ``registration.scope`` are silently skipped.  Per-file parse failures
        are accumulated in :attr:`IngestResult.errors`; they do not abort the
        run.
        """
        t_start = time.monotonic()
        errors: list[str] = []
        all_mutations: list[TemporalMutation] = []
        files_parsed = 0
        entities_created = 0
        relationships_created = 0
        unresolved_count = 0

        log.info(
            "pipeline.ingest.start",
            repo=registration.name,
        )

        provider = create_git_provider(registration.git_provider)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Step 1: Clone the repository.
            local_path = await provider.clone(
                repo_id=registration.name,
                target_path=tmp_dir,
                branch=registration.git_provider.default_branch,
            )

            # Attempt to capture the HEAD SHA for temporal metadata.
            head_sha = _get_head_sha(local_path)

            # Step 2: Enumerate all files in the local clone.
            root = Path(local_path)
            all_file_paths: list[Path] = [
                p for p in root.rglob("*") if p.is_file()
            ]

            # Step 3 / 4: Filter and select parser per file.
            for abs_path in all_file_paths:
                rel_path = str(abs_path.relative_to(root)).replace("\\", "/")

                if _is_ignored(rel_path, registration.ignore_patterns):
                    continue

                if not _is_in_scope(rel_path, registration):
                    continue

                parser = self._parser_for(rel_path)
                if parser is None:
                    continue  # No parser for this extension → skip silently.

                # Step 5: Parse the file.
                try:
                    content = abs_path.read_bytes()
                    result = await parser.parse_file(rel_path, content)
                except Exception as exc:
                    msg = f"parse_failed:{rel_path}:{exc}"
                    log.warning(
                        "pipeline.parse_error",
                        repo=registration.name,
                        file=rel_path,
                        error=str(exc),
                    )
                    errors.append(msg)
                    continue

                files_parsed += 1

                # Step 6: Convert to TemporalMutation objects.
                for entity in result.entities:
                    all_mutations.append(
                        _entity_to_mutation(entity, registration.name)
                    )
                    entities_created += 1

                for rel in result.relationships:
                    all_mutations.append(
                        _relationship_to_mutation(rel, registration.name)
                    )
                    relationships_created += 1

                unresolved_count += len(result.unresolved_references)

            # Step 7: Apply mutations in batches of _BATCH_SIZE.
            for i in range(0, max(len(all_mutations), 1), _BATCH_SIZE):
                batch = all_mutations[i : i + _BATCH_SIZE]
                if not batch:
                    break
                await graph.apply_mutations(
                    batch,
                    change_source=ChangeSource.INITIAL_INGESTION,
                    commit_sha=head_sha,
                )

        duration = time.monotonic() - t_start
        log.info(
            "pipeline.ingest.done",
            repo=registration.name,
            files_parsed=files_parsed,
            entities_created=entities_created,
            relationships_created=relationships_created,
            duration_seconds=round(duration, 3),
        )

        return IngestResult(
            files_parsed=files_parsed,
            entities_created=entities_created,
            relationships_created=relationships_created,
            unresolved_references=unresolved_count,
            duration_seconds=duration,
            errors=errors,
        )
