"""Phase 2 integration tests — exit-criteria validation.

Tests EC1–EC5 for Phase 2 of the Tractable agent runtime.  All tests require
the Docker Compose stack to be running::

    docker compose -f deploy/docker-compose.yml up -d

Each test skips with a descriptive message when the required service is
unreachable.  No live GitHub API calls are made — all git operations are
performed against local fixture repositories.

Fixture repos
-------------
- ``tests/fixtures/fixture_python_api.tar.gz``  — small Python project
- ``tests/fixtures/fixture_typescript_frontend.tar.gz`` — TypeScript project
  (plain functions/classes only; no interfaces, generics, or decorators)

Usage::

    # Run all EC tests:
    uv run pytest tests/integration/test_phase2_exit_criteria.py

    # Run a single EC:
    uv run pytest tests/integration/test_phase2_exit_criteria.py::test_task_to_pr
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import tarfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog

from tractable.agent.state import AgentWorkflowState
from tractable.agent.workflow import resume_task
from tractable.protocols.tool import ToolResult
from tractable.state.store import PostgreSQLAgentStateStore
from tractable.types.agent import AgentContext, AuditEntry
from tractable.types.enums import TaskPhase
from tractable.types.graph import RepoGraphSummary

# ── Constants ─────────────────────────────────────────────────────────────────

_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://tractable:tractable_dev@localhost:5433/tractable",
)
_TRACTABLE_URL = os.environ.get("TRACTABLE_URL", "http://localhost:8000")
_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

pytestmark = pytest.mark.integration


# ── Skip helpers ──────────────────────────────────────────────────────────────


def _skip_if_postgres_unavailable() -> None:
    """Skip the test if PostgreSQL is not reachable."""
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(_DATABASE_URL, pool_pre_ping=True)
    try:
        asyncio.get_event_loop().run_until_complete(engine.connect().__aenter__())
    except Exception:
        pytest.skip(
            f"PostgreSQL not reachable at {_DATABASE_URL}. "
            "Start the stack: docker compose -f deploy/docker-compose.yml up -d"
        )
    finally:
        asyncio.get_event_loop().run_until_complete(engine.dispose())


def _skip_if_tractable_service_unavailable() -> None:
    """Skip the test if the tractable service is not reachable."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{_TRACTABLE_URL}/health", timeout=3) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
    except Exception:
        pytest.skip(
            f"Tractable service not reachable at {_TRACTABLE_URL}. "
            "Start the stack: docker compose -f deploy/docker-compose.yml up --wait"
        )


# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def state_store() -> PostgreSQLAgentStateStore:
    """Live PostgreSQLAgentStateStore connected to the integration database."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    engine = create_async_engine(_DATABASE_URL, pool_pre_ping=True)
    factory: async_sessionmaker[AsyncSession] = async_sessionmaker(engine, expire_on_commit=False)
    return PostgreSQLAgentStateStore(factory)


def _mock_graph(entities_count: int = 1) -> object:
    """Return a mock CodeGraph that returns a non-empty RepoGraphSummary."""
    summary = RepoGraphSummary(
        repo_name="fixture-repo",
        total_entities=entities_count,
        summary_text="Fixture repository for integration tests.",
    )
    graph = AsyncMock()
    graph.get_repo_summary = AsyncMock(return_value=summary)
    graph.query_current = AsyncMock(return_value=[])
    graph.get_neighborhood = AsyncMock(return_value=MagicMock(entities=[], edges=[]))
    return graph


def _make_agent_context(
    agent_id: str,
    repo: str = "fixture-repo",
    status: str = "DORMANT",
) -> AgentContext:
    return AgentContext(
        agent_id=agent_id,
        repo=repo,
        base_template="api_maintainer",
        system_prompt=f"Agent for {repo}.",
        repo_architectural_summary="",
        user_overrides={"status": status},
    )


# ── EC1: test_task_to_pr ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_task_to_pr(state_store: PostgreSQLAgentStateStore) -> None:
    """EC1: Submit a bug-fix task; verify workflow opens a PR and completes.

    Uses mock LLM (llm_call callback) and mock git_ops tool.
    No live GitHub API calls are made.
    """
    _skip_if_postgres_unavailable()

    agent_id = f"ec1-agent-{uuid.uuid4().hex[:8]}"
    task_id = f"ec1-task-{uuid.uuid4().hex[:8]}"
    pr_url = f"https://github.com/fixture/fixture-python-api/pull/{uuid.uuid4().hex[:4]}"

    # Persist agent context so state_store.get_agent_context succeeds if called.
    ctx = _make_agent_context(agent_id, repo="fixture-python-api")
    await state_store.save_agent_context(agent_id, ctx)

    # Mock git_ops: any invoke returns a deterministic PR URL.
    class _MockGitOpsTool:
        @property
        def name(self) -> str:
            return "git_ops"

        @property
        def description(self) -> str:
            return "Mock git ops"

        async def invoke(self, params: dict[str, object]) -> ToolResult:  # noqa: ARG002
            return ToolResult(success=True, output=pr_url)

    tools: dict[str, object] = {"git_ops": _MockGitOpsTool()}

    # llm_call simulates 1 000 tokens per node — well below the default budget.
    def _llm_call(model: str) -> int:  # noqa: ARG001
        return 1_000

    result = await resume_task(
        agent_id=agent_id,
        task_id=task_id,
        task_description="Fix the zero-division bug in calculator.divide()",
        state_store=state_store,
        tools=tools,  # type: ignore[arg-type]
        graph=_mock_graph(),  # type: ignore[arg-type]
        llm_call=_llm_call,
    )

    # EC1 assertions
    assert result["phase"] == TaskPhase.COMPLETED, f"Expected COMPLETED, got {result['phase']}"
    assert result["pr_url"] == pr_url, f"Expected pr_url={pr_url!r}, got {result['pr_url']!r}"


# ── EC2: test_checkpoint_resume ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_checkpoint_resume(state_store: PostgreSQLAgentStateStore) -> None:
    """EC2: Verify checkpoint restore skips PLANNING on second run.

    Runs the workflow once (saving a PLANNING checkpoint), then reconstructs
    the workflow and calls resume_task again.  Counts PLANNING node invocations:
    must be exactly 1 (not 2 — PLANNING is skipped on restore).
    """
    _skip_if_postgres_unavailable()

    agent_id = f"ec2-agent-{uuid.uuid4().hex[:8]}"
    task_id = f"ec2-task-{uuid.uuid4().hex[:8]}"

    ctx = _make_agent_context(agent_id)
    await state_store.save_agent_context(agent_id, ctx)

    planning_call_count = 0

    from tractable.agent.nodes import plan as _plan_module  # noqa: PLC0415

    original_make_planning = _plan_module.make_planning_node

    def _counting_make_planning(
        tools: dict[str, object],
        ss: object,
        g: object,
    ) -> object:
        original = original_make_planning(tools, ss, g)  # type: ignore[arg-type]

        async def wrapper(state: AgentWorkflowState) -> dict[str, Any]:
            nonlocal planning_call_count
            planning_call_count += 1
            return await original(state)

        return wrapper

    def _llm_call(model: str) -> int:  # noqa: ARG001
        return 500

    with patch.object(_plan_module, "make_planning_node", _counting_make_planning):
        # First run — PLANNING executes and saves a checkpoint.
        await resume_task(
            agent_id=agent_id,
            task_id=task_id,
            task_description="Fix the bug",
            state_store=state_store,
            tools={},
            graph=_mock_graph(),  # type: ignore[arg-type]
            llm_call=_llm_call,
        )

    first_run_count = planning_call_count

    # Verify a PLANNING checkpoint was saved.
    checkpoint = await state_store.get_checkpoint(agent_id, task_id)
    assert checkpoint is not None, "No checkpoint found after first run"

    # Second run — simulate process restart by reconstructing workflow and
    # loading the existing checkpoint.  PLANNING should NOT run again.
    planning_call_count = 0

    with (
        patch.object(_plan_module, "make_planning_node", _counting_make_planning),
        structlog.testing.capture_logs() as captured_logs,
    ):
        await resume_task(
            agent_id=agent_id,
            task_id=task_id,
            task_description="Fix the bug",
            state_store=state_store,
            tools={},
            graph=_mock_graph(),  # type: ignore[arg-type]
            llm_call=_llm_call,
        )

    second_run_count = planning_call_count

    # EC2 assertions
    assert first_run_count == 1, (
        f"Expected exactly 1 PLANNING call in first run; got {first_run_count}"
    )
    assert second_run_count == 0, (
        f"Expected 0 PLANNING calls on restore (checkpoint resume); got {second_run_count}"
    )

    checkpoint_events = [r for r in captured_logs if r.get("event") == "checkpoint_restored"]
    assert checkpoint_events, "Expected 'checkpoint_restored' in structlog output on second run"


# ── EC3: test_agent_wake_on_webhook ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_wake_on_webhook(state_store: PostgreSQLAgentStateStore) -> None:
    """EC3: POST webhook to tractable service; verify agent wakes in PostgreSQL.

    Requires the tractable service running at localhost:8000.
    Polls for AgentContext.last_active update for up to 35 seconds.
    """
    _skip_if_postgres_unavailable()
    _skip_if_tractable_service_unavailable()

    import urllib.request

    agent_id = f"ec3-agent-{uuid.uuid4().hex[:8]}"
    repo_name = f"fixture-ec3-{uuid.uuid4().hex[:6]}"

    # Register the agent in PostgreSQL with DORMANT status and no last_active.
    ctx = _make_agent_context(agent_id, repo=repo_name, status="DORMANT")
    ctx.user_overrides.pop("last_active", None)
    await state_store.save_agent_context(agent_id, ctx)

    # Build a minimal GitHub push webhook payload for the fixture repo.
    payload = {
        "ref": "refs/heads/main",
        "before": "0" * 40,
        "after": "a" * 40,
        "repository": {
            "full_name": repo_name,
            "pushed_at": int(datetime.now(tz=UTC).timestamp()),
        },
        "pusher": {"name": "test-user"},
        "commits": [
            {
                "id": "a" * 40,
                "message": "Fix the bug",
                "author": {"name": "test-user"},
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "added": [],
                "modified": ["src/calculator.py"],
                "removed": [],
            }
        ],
    }
    body = json.dumps(payload).encode()

    # Sign with the configured webhook secret (default: empty string).
    webhook_secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "")
    sig = "sha256=" + hmac.new(webhook_secret.encode(), body, hashlib.sha256).hexdigest()

    req = urllib.request.Request(
        f"{_TRACTABLE_URL}/webhooks/github",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": "push",
            "X-GitHub-Delivery": f"test-delivery-{uuid.uuid4().hex[:8]}",
            "X-Hub-Signature-256": sig,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status in (200, 202), f"Unexpected status {resp.status}"

    # Poll for last_active to be set (up to 35 seconds).
    deadline = asyncio.get_event_loop().time() + 35.0
    last_active: str | None = None
    while asyncio.get_event_loop().time() < deadline:
        refreshed = await state_store.get_agent_context(agent_id)
        last_active = str(refreshed.user_overrides.get("last_active", ""))
        if last_active:
            break
        await asyncio.sleep(0.5)

    # EC3 assertions — primary: last_active updated in PostgreSQL.
    assert last_active, (
        "AgentContext.last_active was not updated within 35 seconds after webhook POST. "
        "Ensure the tractable service can reach PostgreSQL and the agent repo matches."
    )

    # EC3 assertions — secondary: verify agent_woke event via in-process log capture.
    from tractable.agent.lifecycle import LifecycleManager  # noqa: PLC0415
    from tractable.graph.client import FalkorDBClient  # noqa: PLC0415
    from tractable.graph.temporal_graph import FalkorDBTemporalGraph  # noqa: PLC0415

    _falkordb_host = os.environ.get("FALKORDB_HOST", "localhost")
    _falkordb_port = int(os.environ.get("FALKORDB_PORT", "6380"))
    client = FalkorDBClient(
        host=_falkordb_host, port=_falkordb_port, graph_name="tractable_integration_test"
    )
    inproc_graph = FalkorDBTemporalGraph(client)

    manager = LifecycleManager(
        state_store=state_store,
        graph=inproc_graph,
        registrations={},
        working_dirs={},
    )

    # Create a fresh agent for the in-process wake test.
    ip_agent_id = f"ec3-inproc-{uuid.uuid4().hex[:8]}"
    ip_ctx = _make_agent_context(ip_agent_id, repo=repo_name, status="DORMANT")
    await state_store.save_agent_context(ip_agent_id, ip_ctx)

    with structlog.testing.capture_logs() as captured_logs:
        await manager.wake_agent(ip_agent_id, "test")

    wake_events = [r for r in captured_logs if r.get("event") == "agent_woke"]
    assert wake_events, "Expected event='agent_woke' in structlog output from wake_agent()"

    await client.close()


# ── EC4: test_audit_log_completeness ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_audit_log_completeness(
    state_store: PostgreSQLAgentStateStore,
) -> None:
    """EC4: Run full workflow; verify audit log covers required action types.

    Required actions: file_written, graph_query, pull_request_created.
    Uses mock tools that emit audit entries for each required action type.
    """
    _skip_if_postgres_unavailable()

    agent_id = f"ec4-agent-{uuid.uuid4().hex[:8]}"
    task_id = f"ec4-task-{uuid.uuid4().hex[:8]}"

    ctx = _make_agent_context(agent_id)
    await state_store.save_agent_context(agent_id, ctx)

    # ── Mock tools that append audit entries ──────────────────────────────

    class _AuditingCodeEditor:
        @property
        def name(self) -> str:
            return "code_editor"

        @property
        def description(self) -> str:
            return "Mock code editor that emits audit entries"

        async def invoke(self, params: dict[str, object]) -> ToolResult:  # noqa: ARG002
            await state_store.append_audit_entry(
                AuditEntry(
                    timestamp=datetime.now(tz=UTC),
                    agent_id=agent_id,
                    task_id=task_id,
                    action="file_written",
                    detail={"path": "src/calculator.py", "bytes_written": 256},
                    outcome="success",
                )
            )
            return ToolResult(success=True, output="src/calculator.py")

    class _AuditingGitOps:
        @property
        def name(self) -> str:
            return "git_ops"

        @property
        def description(self) -> str:
            return "Mock git ops that emits audit entries"

        async def invoke(self, params: dict[str, object]) -> ToolResult:  # noqa: ARG002
            await state_store.append_audit_entry(
                AuditEntry(
                    timestamp=datetime.now(tz=UTC),
                    agent_id=agent_id,
                    task_id=task_id,
                    action="pull_request_created",
                    detail={"url": "https://github.com/fixture/pr/1"},
                    outcome="success",
                )
            )
            return ToolResult(
                success=True,
                output="https://github.com/fixture/pr/1",
            )

    # Emit a graph_query audit entry from the mock graph's get_repo_summary.
    async def _auditing_get_repo_summary(desc: str) -> RepoGraphSummary:  # noqa: ARG001
        await state_store.append_audit_entry(
            AuditEntry(
                timestamp=datetime.now(tz=UTC),
                agent_id=agent_id,
                task_id=task_id,
                action="graph_query",
                detail={"cypher": "MATCH (e:Entity) RETURN e LIMIT 10"},
                outcome="success",
            )
        )
        return RepoGraphSummary(
            repo_name="fixture-repo",
            total_entities=3,
            summary_text="Graph queried.",
        )

    mock_graph = _mock_graph()
    mock_graph.get_repo_summary = _auditing_get_repo_summary  # type: ignore[attr-defined]

    tools: dict[str, object] = {
        "code_editor": _AuditingCodeEditor(),
        "git_ops": _AuditingGitOps(),
    }

    await resume_task(
        agent_id=agent_id,
        task_id=task_id,
        task_description="Fix calculator divide bug",
        state_store=state_store,
        tools=tools,  # type: ignore[arg-type]
        graph=mock_graph,  # type: ignore[arg-type]
        llm_call=lambda model: 500,  # noqa: ARG005
    )

    # ── Query audit log and verify action coverage ────────────────────────
    entries = await state_store.get_audit_log(task_id=task_id)
    actions = {e.action for e in entries}

    assert "file_written" in actions, f"Expected 'file_written' in audit actions; got {actions}"
    assert "graph_query" in actions, f"Expected 'graph_query' in audit actions; got {actions}"
    assert "pull_request_created" in actions, (
        f"Expected 'pull_request_created' in audit actions; got {actions}"
    )


# ── EC5: test_register_two_repos ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_register_two_repos(
    state_store: PostgreSQLAgentStateStore,
    tmp_path: Path,
) -> None:
    """EC5: Register Python and TypeScript fixture repos; verify agents in list.

    Extracts fixture tarballs, mocks git provider clone to return the local
    extracted paths, runs GraphConstructionPipeline for each repo, saves
    AgentContext to the state store, and verifies both agents appear in
    tractable agent list with status DORMANT.

    TypeScript fixture is constrained to plain functions/classes only — no
    interfaces, generics, or decorators (Phase 4 deferred constructs).
    """
    _skip_if_postgres_unavailable()

    py_tarball = _FIXTURES_DIR / "fixture_python_api.tar.gz"
    ts_tarball = _FIXTURES_DIR / "fixture_typescript_frontend.tar.gz"

    if not py_tarball.exists():
        pytest.skip(f"Python fixture tarball not found: {py_tarball}")
    if not ts_tarball.exists():
        pytest.skip(f"TypeScript fixture tarball not found: {ts_tarball}")

    # ── Extract fixtures ──────────────────────────────────────────────────
    py_extract = tmp_path / "python"
    ts_extract = tmp_path / "typescript"
    py_extract.mkdir()
    ts_extract.mkdir()

    with tarfile.open(py_tarball) as tar:
        tar.extractall(py_extract)
    with tarfile.open(ts_tarball) as tar:
        tar.extractall(ts_extract)

    py_src = py_extract / "fixture_python_api"
    ts_src = ts_extract / "fixture_typescript_frontend"

    # ── Unique repo names to avoid cross-test contamination ───────────────
    run_id = uuid.uuid4().hex[:8]
    py_repo = f"fixture-python-api-{run_id}"
    ts_repo = f"fixture-typescript-frontend-{run_id}"
    py_agent_id = f"{py_repo}-agent"
    ts_agent_id = f"{ts_repo}-agent"

    from tractable.graph.client import FalkorDBClient  # noqa: PLC0415
    from tractable.graph.temporal_graph import FalkorDBTemporalGraph  # noqa: PLC0415
    from tractable.parsing.parsers.typescript_parser import TypeScriptParser  # noqa: PLC0415
    from tractable.parsing.pipeline import GraphConstructionPipeline  # noqa: PLC0415
    from tractable.types.config import GitProviderConfig, RepositoryRegistration  # noqa: PLC0415

    _falkordb_host = os.environ.get("FALKORDB_HOST", "localhost")
    _falkordb_port = int(os.environ.get("FALKORDB_PORT", "6380"))
    client = FalkorDBClient(
        host=_falkordb_host,
        port=_falkordb_port,
        graph_name=f"tractable_integration_ec5_{run_id}",
    )
    inproc_graph = FalkorDBTemporalGraph(client)

    py_reg = RepositoryRegistration(
        name=py_repo,
        git_url="https://github.com/fixture/python-api.git",
        git_provider=GitProviderConfig(
            provider_type="github",
            credentials_secret_ref="GITHUB_TOKEN",
        ),
        primary_language="python",
    )
    ts_reg = RepositoryRegistration(
        name=ts_repo,
        git_url="https://github.com/fixture/typescript-frontend.git",
        git_provider=GitProviderConfig(
            provider_type="github",
            credentials_secret_ref="GITHUB_TOKEN",
        ),
        primary_language="typescript",
    )

    # ── Ingest Python fixture (mock clone to return local extracted dir) ──
    import tractable.parsing.pipeline as _pipeline_mod  # noqa: PLC0415

    py_pipeline = GraphConstructionPipeline()
    with patch.object(
        _pipeline_mod,
        "create_git_provider",
        return_value=_mock_clone_provider(str(py_src)),
    ):
        py_result = await py_pipeline.ingest_repository(py_reg, inproc_graph)

    # ── Ingest TypeScript fixture ─────────────────────────────────────────
    ts_pipeline = GraphConstructionPipeline(extra_parsers=[TypeScriptParser()])
    with patch.object(
        _pipeline_mod,
        "create_git_provider",
        return_value=_mock_clone_provider(str(ts_src)),
    ):
        ts_result = await ts_pipeline.ingest_repository(ts_reg, inproc_graph)

    # ── Save AgentContexts with DORMANT status ────────────────────────────
    py_ctx = _make_agent_context(py_agent_id, repo=py_repo, status="DORMANT")
    ts_ctx = _make_agent_context(ts_agent_id, repo=ts_repo, status="DORMANT")
    await state_store.save_agent_context(py_agent_id, py_ctx)
    await state_store.save_agent_context(ts_agent_id, ts_ctx)

    # ── EC5 assertions ────────────────────────────────────────────────────
    assert py_result.entities_created > 0, (
        f"Python fixture should have >0 entities; got {py_result.entities_created}. "
        f"Errors: {py_result.errors}"
    )
    assert ts_result.entities_created > 0, (
        f"TypeScript fixture should have >0 entities; got {ts_result.entities_created}. "
        f"Errors: {ts_result.errors}"
    )

    agents = await state_store.list_agents()
    agent_map = {a.agent_id: a for a in agents}

    assert py_agent_id in agent_map, (
        f"Python agent {py_agent_id!r} not found in agent list. Present agents: {list(agent_map)}"
    )
    assert ts_agent_id in agent_map, (
        f"TypeScript agent {ts_agent_id!r} not found in agent list. "
        f"Present agents: {list(agent_map)}"
    )

    for aid in (py_agent_id, ts_agent_id):
        status = agent_map[aid].user_overrides.get("status")
        assert status == "DORMANT", f"Agent {aid!r} expected status DORMANT; got {status!r}"

    await client.close()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mock_clone_provider(local_path: str) -> object:
    """Return a mock GitProvider whose clone() returns *local_path* directly.

    The pipeline's ``ingest_repository`` calls::

        local_path = await provider.clone(repo_id=..., target_path=tmp_dir, ...)

    We override ``clone`` to ignore *target_path* and return the pre-extracted
    fixture directory path, allowing the pipeline to parse files from the
    fixture without any network access.
    """
    mock = MagicMock()

    async def _clone(
        repo_id: str,  # noqa: ARG001
        target_path: str,  # noqa: ARG001
        branch: str = "main",  # noqa: ARG001
        **kw: object,  # noqa: ARG002
    ) -> str:
        return local_path

    mock.clone = _clone
    return mock
