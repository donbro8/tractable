# pyright: reportUnusedClass=false
"""ChangeIngestionPipeline implementation.

TASK-2.6.1 — Implement FastAPI webhook receiver and ChangeIngestionPipeline.

``GitChangeIngestionPipeline`` satisfies the ``ChangeIngestionPipeline`` Protocol
(tractable/protocols/reactivity.py).  It:

1. Checks Redis for a duplicate event_id (idempotency, 24-hour TTL).
2. Collects all affected file paths from the event's commits.
3. For each non-removed file: fetches content via ``GitProvider`` and re-parses
   with the appropriate ``CodeParser``.
4. Produces ``TemporalMutation`` objects (create/update/delete) by diffing parse
   results against the current graph.
5. Calls ``TemporalCodeGraph.apply_mutations()`` with
   ``ChangeSource.INCREMENTAL_UPDATE``.
6. Logs ``event="graph_updated"`` and returns a ``ChangeIngestionResult``.

On parse failure for a single file: logs a warning and continues (non-fatal).
On graph mutation failure: raises ``TransientError``.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

import structlog

from tractable.errors import TransientError
from tractable.protocols.code_graph import TemporalCodeGraph
from tractable.protocols.git_provider import GitProvider
from tractable.protocols.graph_construction import CodeParser, ParseResult
from tractable.protocols.reactivity import (
    ChangeIngestionResult,
    RepositoryChangeEvent,
)
from tractable.types.enums import ChangeSource
from tractable.types.temporal import TemporalMutation, TemporalMutationResult

_log = structlog.get_logger()


class _AsyncRedis(Protocol):
    """Minimal async Redis protocol used for idempotency key management."""

    def get(self, key: str) -> Awaitable[bytes | None]:
        ...

    def set(self, key: str, value: str, *, ex: int) -> Awaitable[Any]:
        ...

# Redis key template for processed event idempotency.
_REDIS_KEY_PREFIX = "tractable:event:"
_REDIS_TTL_SECONDS = 86_400  # 24 hours


class GitChangeIngestionPipeline:
    """Concrete ``ChangeIngestionPipeline`` that ingests git push events.

    Parameters
    ----------
    git_provider:
        Used to fetch file content at ``event.after_sha``.
    graph:
        Temporal code graph; used to query current entities and apply mutations.
    parsers:
        Sequence of ``CodeParser`` objects.  The pipeline selects the first
        parser whose ``supported_extensions`` contains the file's suffix.
    redis_client:
        An async Redis client (``redis.asyncio.Redis``).  Used exclusively for
        idempotency key management.  Must expose ``get(key)``, ``set(key, value,
        ex=ttl)`` coroutines.
    """

    def __init__(
        self,
        git_provider: GitProvider,
        graph: TemporalCodeGraph,
        parsers: Sequence[CodeParser],
        redis_client: _AsyncRedis,
    ) -> None:
        self._git_provider = git_provider
        self._graph = graph
        self._parsers = list(parsers)
        self._redis = redis_client

    # ── Public API (satisfies ChangeIngestionPipeline Protocol) ───────────

    async def process_change(
        self,
        event: RepositoryChangeEvent,
    ) -> ChangeIngestionResult:
        """Process a single repository change event end-to-end.

        Returns a ``ChangeIngestionResult`` with ``files_modified=0`` and no
        mutations when the event has already been processed (idempotency).
        Raises ``TransientError`` if the graph mutation call fails.
        """
        # ── Step 1: Idempotency check ──────────────────────────────────────
        redis_key = f"{_REDIS_KEY_PREFIX}{event.event_id}"
        if await self._redis.get(redis_key) is not None:
            _log.info(
                "duplicate_event_skipped",
                event_id=event.event_id,
                repo=event.repo_name,
            )
            _noop_result = TemporalMutationResult(
                entities_created=0,
                entities_updated=0,
                entities_deleted=0,
                edges_created=0,
                edges_deleted=0,
                errors=[],
                timestamp=datetime.now(tz=UTC),
            )
            return ChangeIngestionResult(
                event_id=event.event_id,
                repo_name=event.repo_name,
                commit_sha=event.after_sha,
                files_added=0,
                files_modified=0,
                files_removed=0,
                parse_duration_ms=0,
                graph_mutations=_noop_result,
            )

        # ── Step 2: Collect affected files ─────────────────────────────────
        added_files: set[str] = set()
        modified_files: set[str] = set()
        removed_files: set[str] = set()

        for commit in event.commits:
            added_files.update(commit.added_files)
            modified_files.update(commit.modified_files)
            removed_files.update(commit.removed_files)

        # Files that were later removed in the same push: treat as removed only.
        added_files -= removed_files
        modified_files -= removed_files

        # ── Step 3: Fetch + parse non-removed files ────────────────────────
        mutations: list[TemporalMutation] = []
        warnings: list[str] = []

        parse_start_ns = time.monotonic_ns()

        for file_path in added_files | modified_files:
            parser = self._find_parser(file_path)
            if parser is None:
                continue  # No parser for this extension — skip silently.

            try:
                content: bytes = await self._git_provider.get_file_content(
                    event.repo_name,
                    file_path,
                    ref=event.after_sha,
                )
                parse_result: ParseResult = await parser.parse_file(file_path, content)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"parse_failed:{file_path}:{exc}")
                _log.warning(
                    "file_parse_failed",
                    repo=event.repo_name,
                    file_path=file_path,
                    error=str(exc),
                )
                continue

            mutations.extend(self._mutations_from_parse(parse_result, file_path))

        # Generate delete mutations for removed files.
        for file_path in removed_files:
            mutations.append(
                TemporalMutation(
                    operation="delete_entity",
                    payload={"file_path": file_path, "repo": event.repo_name},
                )
            )

        parse_duration_ms = (time.monotonic_ns() - parse_start_ns) // 1_000_000

        # ── Step 4: Apply mutations ────────────────────────────────────────
        try:
            mutation_result: TemporalMutationResult = await self._graph.apply_mutations(
                mutations,
                change_source=ChangeSource.INCREMENTAL_UPDATE,
                commit_sha=event.after_sha,
            )
        except Exception as exc:
            raise TransientError(
                f"Graph mutation failed for event {event.event_id}: {exc}"
            ) from exc

        # ── Step 5: Mark event as processed in Redis ──────────────────────
        await self._redis.set(redis_key, "1", ex=_REDIS_TTL_SECONDS)

        _log.info(
            "graph_updated",
            repo=event.repo_name,
            commit_sha=event.after_sha,
            files_changed=len(added_files) + len(modified_files) + len(removed_files),
            mutations_applied=(
                mutation_result.entities_created
                + mutation_result.entities_updated
                + mutation_result.entities_deleted
            ),
        )

        return ChangeIngestionResult(
            event_id=event.event_id,
            repo_name=event.repo_name,
            commit_sha=event.after_sha,
            files_added=len(added_files),
            files_modified=len(modified_files),
            files_removed=len(removed_files),
            parse_duration_ms=parse_duration_ms,
            graph_mutations=mutation_result,
            warnings=warnings,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _find_parser(self, file_path: str) -> CodeParser | None:
        """Return the first parser that supports this file's extension."""
        suffix = "." + file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        for parser in self._parsers:
            if suffix in parser.supported_extensions:
                return parser
        return None

    def _mutations_from_parse(
        self,
        parse_result: ParseResult,
        file_path: str,
    ) -> list[TemporalMutation]:
        """Convert parsed entities into ``TemporalMutation`` objects.

        Each parsed entity becomes an ``update_entity`` mutation so the
        temporal graph can close the previous version and open a new one.
        The graph implementation distinguishes create vs update internally
        based on whether the entity_id already exists.
        """
        mutations: list[TemporalMutation] = []
        for entity in parse_result.entities:
            mutations.append(
                TemporalMutation(
                    operation="update_entity",
                    entity_id=entity.qualified_name,
                    payload={
                        "kind": entity.kind,
                        "name": entity.name,
                        "qualified_name": entity.qualified_name,
                        "file_path": file_path,
                        "line_start": entity.line_start,
                        "line_end": entity.line_end,
                        "properties": entity.properties,
                    },
                )
            )
        return mutations
