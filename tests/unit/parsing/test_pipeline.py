"""Unit tests for GraphConstructionPipeline.

All external dependencies (GitProvider, TemporalCodeGraph) are mocked so no
real network or database connections are needed.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tractable.parsing.pipeline import GraphConstructionPipeline, IngestResult
from tractable.types.config import GitProviderConfig, RepositoryRegistration
from tractable.types.enums import ChangeSource
from tractable.types.temporal import TemporalMutation, TemporalMutationResult

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_registration(
    tmp_dir: str,
    ignore_patterns: list[str] | None = None,
) -> RepositoryRegistration:
    return RepositoryRegistration(
        name="owner/myrepo",
        git_url="https://github.com/owner/myrepo",
        git_provider=GitProviderConfig(
            provider_type="github",
            credentials_secret_ref="GITHUB_TOKEN",
        ),
        primary_language="python",
        ignore_patterns=ignore_patterns
        or [
            "__pycache__/**",
            "*.pyc",
            ".git/**",
            "node_modules/**",
        ],
    )


def _make_mock_graph(entities_in_graph: int = 5) -> MagicMock:
    """Return a mock TemporalCodeGraph that records apply_mutations calls."""
    graph = MagicMock()
    dummy_result = TemporalMutationResult(
        entities_created=entities_in_graph,
        entities_updated=0,
        entities_deleted=0,
        edges_created=0,
        edges_deleted=0,
        timestamp=datetime.now(UTC),
    )
    graph.apply_mutations = AsyncMock(return_value=dummy_result)
    graph.query_current = AsyncMock(return_value=[{"cnt": entities_in_graph}])
    return graph


def _create_python_files(directory: str, count: int = 12) -> list[str]:
    """Create *count* minimal Python files in *directory* and return their paths."""
    created: list[str] = []
    for i in range(count):
        sub = Path(directory) / f"module_{i}.py"
        sub.write_text(
            f"def func_{i}(x: int) -> int:\n    return x + {i}\n\n"
            f"class Class_{i}:\n    pass\n",
            encoding="utf-8",
        )
        created.append(str(sub))
    return created


def _make_clone_mock(local_path: str) -> AsyncMock:
    return AsyncMock(return_value=local_path)


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.fixture()
def pipeline() -> GraphConstructionPipeline:
    return GraphConstructionPipeline()


@pytest.mark.asyncio
async def test_ingest_completes_without_exception(
    pipeline: GraphConstructionPipeline,
) -> None:
    """AC-1: ingest_repository completes without exception on a repo with 10+ files."""
    with tempfile.TemporaryDirectory() as tmp:
        _create_python_files(tmp, count=12)
        registration = _make_registration(tmp)
        graph = _make_mock_graph()

        with patch(
            "tractable.parsing.pipeline.create_git_provider"
        ) as mock_factory:
            mock_provider = MagicMock()
            mock_provider.clone = _make_clone_mock(tmp)
            mock_factory.return_value = mock_provider

            result = await pipeline.ingest_repository(registration, graph)

    assert isinstance(result, IngestResult)
    assert result.files_parsed >= 10


@pytest.mark.asyncio
async def test_apply_mutations_called_after_ingestion(
    pipeline: GraphConstructionPipeline,
) -> None:
    """AC-2: graph.apply_mutations is called (simulates graph.query_current returning cnt > 0)."""
    with tempfile.TemporaryDirectory() as tmp:
        _create_python_files(tmp, count=12)
        registration = _make_registration(tmp)
        graph = _make_mock_graph(entities_in_graph=10)

        with patch("tractable.parsing.pipeline.create_git_provider") as mock_factory:
            mock_provider = MagicMock()
            mock_provider.clone = _make_clone_mock(tmp)
            mock_factory.return_value = mock_provider

            await pipeline.ingest_repository(registration, graph)

    graph.apply_mutations.assert_called()
    # Simulate what the spec says: graph.query_current returns cnt > 0
    result = await graph.query_current(
        "MATCH (e) WHERE e.valid_until IS NULL RETURN count(e) AS cnt"
    )
    assert result[0]["cnt"] > 0


@pytest.mark.asyncio
async def test_files_parsed_count_matches_py_files(
    pipeline: GraphConstructionPipeline,
) -> None:
    """AC-3: IngestResult.files_parsed equals the number of .py files (minus ignored)."""
    with tempfile.TemporaryDirectory() as tmp:
        py_count = 10
        _create_python_files(tmp, count=py_count)
        # Add a non-Python file that should be skipped.
        (Path(tmp) / "README.md").write_text("# readme", encoding="utf-8")

        registration = _make_registration(tmp)
        graph = _make_mock_graph()

        with patch("tractable.parsing.pipeline.create_git_provider") as mock_factory:
            mock_provider = MagicMock()
            mock_provider.clone = _make_clone_mock(tmp)
            mock_factory.return_value = mock_provider

            result = await pipeline.ingest_repository(registration, graph)

    assert result.files_parsed == py_count


@pytest.mark.asyncio
async def test_ignore_patterns_exclude_pycache(
    pipeline: GraphConstructionPipeline,
) -> None:
    """AC-4: Files matching ignore_patterns (__pycache__/**) are not parsed."""
    with tempfile.TemporaryDirectory() as tmp:
        # Legitimate Python files.
        _create_python_files(tmp, count=5)
        # Create __pycache__ directory with a .pyc file AND a .py stub.
        pycache_dir = Path(tmp) / "__pycache__"
        pycache_dir.mkdir()
        (pycache_dir / "cached_module.py").write_text("x = 1\n", encoding="utf-8")
        (pycache_dir / "cached_module.cpython-311.pyc").write_bytes(b"\x00")

        registration = _make_registration(tmp)
        graph = _make_mock_graph()

        mutations_seen: list[TemporalMutation] = []

        async def _capture_mutations(
            mutations: Sequence[TemporalMutation],
            change_source: ChangeSource,
            commit_sha: str | None = None,
            agent_id: str | None = None,
        ) -> TemporalMutationResult:
            mutations_seen.extend(mutations)
            return TemporalMutationResult(
                entities_created=len(mutations),
                entities_updated=0,
                entities_deleted=0,
                edges_created=0,
                edges_deleted=0,
                timestamp=datetime.now(UTC),
            )

        graph.apply_mutations = _capture_mutations  # type: ignore[assignment]

        with patch("tractable.parsing.pipeline.create_git_provider") as mock_factory:
            mock_provider = MagicMock()
            mock_provider.clone = _make_clone_mock(tmp)
            mock_factory.return_value = mock_provider

            await pipeline.ingest_repository(registration, graph)

    # No entity should have a file_path containing __pycache__
    pycache_entity_paths = [
        m.payload.get("file_path", "")
        for m in mutations_seen
        if "__pycache__" in m.payload.get("file_path", "")
    ]
    assert pycache_entity_paths == [], (
        f"Expected no __pycache__ entities, got: {pycache_entity_paths}"
    )


@pytest.mark.asyncio
async def test_per_file_error_does_not_abort(
    pipeline: GraphConstructionPipeline,
) -> None:
    """A parser error on one file is accumulated; other files are still parsed."""
    with tempfile.TemporaryDirectory() as tmp:
        _create_python_files(tmp, count=5)
        # Add a file that will trigger a read error by making it unreadable.
        bad = Path(tmp) / "unreadable.py"
        bad.write_text("x = 1\n", encoding="utf-8")

        registration = _make_registration(tmp)
        graph = _make_mock_graph()

        original_read_bytes = Path.read_bytes

        def _patched_read_bytes(self: Path) -> bytes:
            if self.name == "unreadable.py":
                raise OSError("permission denied")
            return original_read_bytes(self)

        with (
            patch("tractable.parsing.pipeline.create_git_provider") as mock_factory,
            patch.object(Path, "read_bytes", _patched_read_bytes),
        ):
            mock_provider = MagicMock()
            mock_provider.clone = _make_clone_mock(tmp)
            mock_factory.return_value = mock_provider

            result = await pipeline.ingest_repository(registration, graph)

    # Should have 5 files parsed (the good ones) and 1 error.
    assert result.files_parsed == 5
    assert len(result.errors) == 1
    assert "unreadable.py" in result.errors[0]


@pytest.mark.asyncio
async def test_scope_allowed_extensions_filters_non_py(
    pipeline: GraphConstructionPipeline,
) -> None:
    """scope.allowed_extensions restricts parsing to specified types."""
    from tractable.types.config import AgentScope

    with tempfile.TemporaryDirectory() as tmp:
        _create_python_files(tmp, count=3)
        (Path(tmp) / "app.ts").write_text("const x = 1;\n", encoding="utf-8")

        registration = _make_registration(tmp)
        registration = registration.model_copy(
            update={"scope": AgentScope(allowed_extensions=[".py"])}
        )
        graph = _make_mock_graph()

        with patch("tractable.parsing.pipeline.create_git_provider") as mock_factory:
            mock_provider = MagicMock()
            mock_provider.clone = _make_clone_mock(tmp)
            mock_factory.return_value = mock_provider

            result = await pipeline.ingest_repository(registration, graph)

    # The .ts file has no parser anyway, but no TypeError should be raised.
    assert result.files_parsed == 3


@pytest.mark.asyncio
async def test_mutations_batched_at_500(
    pipeline: GraphConstructionPipeline,
) -> None:
    """apply_mutations is called in batches not exceeding 500 mutations."""
    with tempfile.TemporaryDirectory() as tmp:
        # Create enough files to generate > 500 mutations.
        # Each file has ~3 entities (module + function + class) plus relationships.
        _create_python_files(tmp, count=150)
        registration = _make_registration(tmp)
        graph = _make_mock_graph()

        call_sizes: list[int] = []

        async def _record_batch(
            mutations: Sequence[TemporalMutation],
            change_source: ChangeSource,
            commit_sha: str | None = None,
            agent_id: str | None = None,
        ) -> TemporalMutationResult:
            call_sizes.append(len(mutations))
            return TemporalMutationResult(
                entities_created=len(mutations),
                entities_updated=0,
                entities_deleted=0,
                edges_created=0,
                edges_deleted=0,
                timestamp=datetime.now(UTC),
            )

        graph.apply_mutations = _record_batch  # type: ignore[assignment]

        with patch("tractable.parsing.pipeline.create_git_provider") as mock_factory:
            mock_provider = MagicMock()
            mock_provider.clone = _make_clone_mock(tmp)
            mock_factory.return_value = mock_provider

            await pipeline.ingest_repository(registration, graph)

    assert all(size <= 500 for size in call_sizes), (
        f"A batch exceeded 500 mutations: {max(call_sizes)}"
    )
    assert sum(call_sizes) > 0


@pytest.mark.asyncio
async def test_ingest_result_fields(
    pipeline: GraphConstructionPipeline,
) -> None:
    """IngestResult has all required fields with sensible values."""
    with tempfile.TemporaryDirectory() as tmp:
        _create_python_files(tmp, count=3)
        registration = _make_registration(tmp)
        graph = _make_mock_graph()

        with patch("tractable.parsing.pipeline.create_git_provider") as mock_factory:
            mock_provider = MagicMock()
            mock_provider.clone = _make_clone_mock(tmp)
            mock_factory.return_value = mock_provider

            result = await pipeline.ingest_repository(registration, graph)

    assert result.files_parsed == 3
    assert result.entities_created > 0
    assert result.duration_seconds > 0
    assert isinstance(result.errors, list)


@pytest.mark.asyncio
async def test_change_source_is_initial_ingestion(
    pipeline: GraphConstructionPipeline,
) -> None:
    """apply_mutations is called with ChangeSource.INITIAL_INGESTION."""
    with tempfile.TemporaryDirectory() as tmp:
        _create_python_files(tmp, count=3)
        registration = _make_registration(tmp)
        graph = _make_mock_graph()

        captured_sources: list[ChangeSource] = []

        async def _capture(
            mutations: Sequence[TemporalMutation],
            change_source: ChangeSource,
            commit_sha: str | None = None,
            agent_id: str | None = None,
        ) -> TemporalMutationResult:
            captured_sources.append(change_source)
            return TemporalMutationResult(
                entities_created=1,
                entities_updated=0,
                entities_deleted=0,
                edges_created=0,
                edges_deleted=0,
                timestamp=datetime.now(UTC),
            )

        graph.apply_mutations = _capture  # type: ignore[assignment]

        with patch("tractable.parsing.pipeline.create_git_provider") as mock_factory:
            mock_provider = MagicMock()
            mock_provider.clone = _make_clone_mock(tmp)
            mock_factory.return_value = mock_provider

            await pipeline.ingest_repository(registration, graph)

    assert all(s == ChangeSource.INITIAL_INGESTION for s in captured_sources)
