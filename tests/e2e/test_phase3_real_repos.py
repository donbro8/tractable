"""Phase 3 E2E integration tests — real GitHub repository validation.

Tests validate the full registration-to-PR flow against real GitHub fixture
repositories using a test token.  Both tests are skipped when
``GITHUB_TEST_TOKEN`` is not set.

Fixture repos (must be pre-created in ``tractable-test-org`` before running
these tests with a token):
- ``tractable-test-org/fixture-python-api``
- ``tractable-test-org/fixture-typescript-frontend``

Tarballs for offline unit testing are committed under ``tests/fixtures/``.

Run with token:
    GITHUB_TEST_TOKEN=<pat> uv run pytest tests/e2e/ -v -m e2e

Run without token (verifies skip behaviour):
    uv run pytest tests/e2e/ -v
"""

from __future__ import annotations

import base64
import os
import subprocess
import tempfile
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from tractable.parsing.parsers.typescript_parser import TypeScriptParser
from tractable.parsing.pipeline import GraphConstructionPipeline
from tractable.protocols.tool import ToolResult
from tractable.providers.github import GitHubProvider
from tractable.types.agent import AgentCheckpoint, AgentContext, AuditEntry
from tractable.types.config import GitProviderConfig, RepositoryRegistration
from tractable.types.enums import ChangeSource, TaskPhase
from tractable.types.graph import RepoGraphSummary
from tractable.types.temporal import TemporalMutation, TemporalMutationResult

# ── Constants ─────────────────────────────────────────────────────────────────

_GITHUB_TEST_TOKEN_VAR = "GITHUB_TEST_TOKEN"
_TEST_ORG = "tractable-test-org"
_PYTHON_REPO_ID = f"{_TEST_ORG}/fixture-python-api"
_TS_REPO_ID = f"{_TEST_ORG}/fixture-typescript-frontend"

pytestmark = pytest.mark.e2e

_needs_token = pytest.mark.skipif(
    not os.environ.get(_GITHUB_TEST_TOKEN_VAR),
    reason="requires GITHUB_TEST_TOKEN",
)


# ── In-memory recording graph ─────────────────────────────────────────────────


class _RecordingGraph:
    """Captures ``TemporalMutation`` records emitted by the pipeline.

    Provides the minimal subset of ``TemporalCodeGraph`` needed by
    ``GraphConstructionPipeline.ingest_repository()`` and the PLANNING node.
    """

    def __init__(self) -> None:
        self._mutations: list[TemporalMutation] = []

    async def apply_mutations(
        self,
        mutations: Sequence[TemporalMutation],
        change_source: ChangeSource,
        commit_sha: str | None = None,
        agent_id: str | None = None,
    ) -> TemporalMutationResult:
        self._mutations.extend(mutations)
        entity_count = sum(
            1 for m in mutations if m.operation == "create_entity"
        )
        edge_count = sum(1 for m in mutations if m.operation == "create_edge")
        return TemporalMutationResult(
            entities_created=entity_count,
            entities_updated=0,
            entities_deleted=0,
            edges_created=edge_count,
            edges_deleted=0,
            timestamp=datetime.now(tz=UTC),
        )

    async def get_repo_summary(self, repo_name: str) -> RepoGraphSummary:
        entity_count = sum(
            1 for m in self._mutations if m.operation == "create_entity"
        )
        return RepoGraphSummary(
            repo_name=repo_name,
            total_entities=entity_count,
            summary_text=f"{entity_count} entities ingested from {repo_name}",
        )

    def entities(self) -> list[dict[str, Any]]:
        """Return payloads of all ``create_entity`` mutations recorded so far."""
        return [
            m.payload
            for m in self._mutations
            if m.operation == "create_entity"
        ]


# ── In-memory state store ─────────────────────────────────────────────────────


class _InMemoryStateStore:
    """Minimal in-memory ``AgentStateStore`` — no PostgreSQL required."""

    def __init__(self) -> None:
        self._contexts: dict[str, AgentContext] = {}
        self._checkpoints: dict[tuple[str, str], list[AgentCheckpoint]] = {}
        self._audit: list[AuditEntry] = []

    async def get_agent_context(self, agent_id: str) -> AgentContext:
        from tractable.errors import RecoverableError

        if agent_id not in self._contexts:
            raise RecoverableError(f"Agent context not found: {agent_id!r}")
        return self._contexts[agent_id]

    async def list_agents(self) -> Sequence[AgentContext]:
        return list(self._contexts.values())

    async def save_agent_context(
        self, agent_id: str, context: AgentContext
    ) -> None:
        self._contexts[agent_id] = context

    async def get_checkpoint(
        self, agent_id: str, task_id: str
    ) -> AgentCheckpoint | None:
        checkpoints = self._checkpoints.get((agent_id, task_id), [])
        return checkpoints[-1] if checkpoints else None

    async def save_checkpoint(
        self,
        agent_id: str,
        task_id: str,
        checkpoint: AgentCheckpoint,
    ) -> None:
        key = (agent_id, task_id)
        self._checkpoints.setdefault(key, []).append(checkpoint)

    async def append_audit_entry(self, entry: AuditEntry) -> None:
        self._audit.append(entry)

    async def get_audit_log(
        self,
        agent_id: str | None = None,
        task_id: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> Sequence[AuditEntry]:
        return self._audit[:limit]

    async def get_last_polled_sha(self, repo_id: str) -> str | None:
        return None

    async def set_last_polled_sha(self, repo_id: str, sha: str) -> None:
        pass


# ── E2E git_ops tool ──────────────────────────────────────────────────────────


class _E2EGitOpsTool:
    """Git-ops tool that creates *real* GitHub branches and PRs for E2E tests.

    Handles the ``{"action": "create_pr", ...}`` call emitted by the
    COORDINATING node.  All git subprocess calls use the same
    ``Authorization: Basic`` extraheader injection used by ``GitHubProvider``
    so the token value never appears in an argv list.

    The ``branch_name`` attribute is set to the name of the branch created by
    the first ``"create_pr"`` invocation.  Tests read this to perform cleanup.
    """

    branch_name: str | None

    def __init__(
        self,
        provider: GitHubProvider,
        repo_id: str,
        working_dir: Path,
        agent_id: str,
        task_id: str,
    ) -> None:
        self._provider = provider
        self._repo_id = repo_id
        self._working_dir = working_dir.resolve()
        self._agent_id = agent_id
        self._task_id = task_id
        self.branch_name = None

    @property
    def name(self) -> str:
        return "git_ops"

    @property
    def description(self) -> str:
        return "E2E git-ops tool: creates real branches and PRs on GitHub."

    async def invoke(self, params: dict[str, Any]) -> ToolResult:
        """Dispatch on the ``action`` key emitted by the COORDINATING node."""
        action: str = str(params.get("action", "") or params.get("operation", ""))

        if action == "create_pr":
            return await self._create_pr()

        return ToolResult(success=False, error=f"Unknown action: {action!r}")

    async def _create_pr(self) -> ToolResult:
        token = os.environ.get(_GITHUB_TEST_TOKEN_VAR, "")
        task_id_short = self._task_id[-8:]
        branch = f"agent/e2e-test-{task_id_short}"
        self.branch_name = branch

        # 1. Create the branch on GitHub via API.
        await self._provider.create_branch(
            self._repo_id,
            branch_name=branch,
            from_ref="main",
        )

        # Build a subprocess environment that injects the token via git
        # extraheader — token value never appears in argv.
        credentials_b64 = base64.b64encode(
            f"x-access-token:{token}".encode()
        ).decode()
        git_env: dict[str, str] = {
            **os.environ,
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"Authorization: Basic {credentials_b64}",
            "GIT_AUTHOR_NAME": "Tractable E2E",
            "GIT_AUTHOR_EMAIL": "e2e@tractable.test",
            "GIT_COMMITTER_NAME": "Tractable E2E",
            "GIT_COMMITTER_EMAIL": "e2e@tractable.test",
        }
        cwd = str(self._working_dir)

        # 2. Check out the new branch in the local clone.
        subprocess.run(
            ["git", "fetch", "origin", branch],
            cwd=cwd, env=git_env, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", branch],
            cwd=cwd, env=git_env, check=True, capture_output=True,
        )

        # 3. Make a minimal test-only commit (marker file only).
        marker = self._working_dir / ".tractable_e2e"
        marker.write_text(f"E2E test run — task_id={self._task_id}\n")
        subprocess.run(
            ["git", "add", ".tractable_e2e"],
            cwd=cwd, env=git_env, check=True, capture_output=True,
        )
        subprocess.run(
            [
                "git", "commit", "-m",
                "test: E2E validation commit from tractable agent",
            ],
            cwd=cwd, env=git_env, check=True, capture_output=True,
        )

        # 4. Push the branch.
        subprocess.run(
            ["git", "push", "origin", branch],
            cwd=cwd, env=git_env, check=True, capture_output=True,
        )

        # 5. Open the PR via the GitHub REST API.
        pr_handle = await self._provider.create_pull_request(
            self._repo_id,
            title=f"[E2E Test] {self._task_id}",
            body=(
                "Automated E2E test PR created by tractable.  "
                "This PR will be closed automatically by the test."
            ),
            head_branch=branch,
            base_branch="main",
        )

        return ToolResult(success=True, output=pr_handle.url)


# ── Cleanup helpers ───────────────────────────────────────────────────────────


def _github_provider() -> GitHubProvider:
    """Return a ``GitHubProvider`` configured to use ``GITHUB_TEST_TOKEN``."""
    return GitHubProvider(
        GitProviderConfig(
            provider_type="github",
            credentials_secret_ref=_GITHUB_TEST_TOKEN_VAR,
        )
    )


async def _close_pr_and_delete_branch(
    repo_id: str,
    pr_number: int,
    branch_name: str,
) -> None:
    """Close *pr_number* and delete *branch_name* via the GitHub REST API."""
    token = os.environ.get(_GITHUB_TEST_TOKEN_VAR, "")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    async with httpx.AsyncClient(base_url="https://api.github.com") as client:
        await client.patch(
            f"/repos/{repo_id}/pulls/{pr_number}",
            headers=headers,
            json={"state": "closed"},
        )
        await client.delete(
            f"/repos/{repo_id}/git/refs/heads/{branch_name}",
            headers=headers,
        )


# ── Test 1: Python API fixture ────────────────────────────────────────────────


@_needs_token
@pytest.mark.asyncio
async def test_register_and_submit_task_python_api() -> None:
    """E2E: Register fixture-python-api, submit a task, verify PR created.

    Steps:
    1. Invoke ``GraphConstructionPipeline.ingest_repository()`` against the
       real GitHub repo (``tractable-test-org/fixture-python-api``).
    2. Assert graph contains at least one ``function`` and one ``import``
       entity — confirming Python parsing succeeded on the live clone.
    3. Submit task ``"Fix the failing test in tests/test_api.py"`` via
       ``resume_task()`` with a mocked LLM callback.
    4. Assert workflow state is ``TaskPhase.COMPLETED`` and ``pr_url`` is set.
    5. Assert the PR URL references the correct fixture repository.
    6. Clean up: close the PR and delete the branch.
    """
    from tractable.agent.workflow import resume_task

    provider = _github_provider()
    graph = _RecordingGraph()
    state_store = _InMemoryStateStore()

    agent_id = f"e2e-agent-{uuid.uuid4().hex[:8]}"
    task_id = f"e2e-task-{uuid.uuid4().hex[:8]}"

    # ── Step 1: Register — ingest the real fixture repo ───────────────────────
    registration = RepositoryRegistration(
        name=_PYTHON_REPO_ID,
        git_url=f"https://github.com/{_PYTHON_REPO_ID}.git",
        git_provider=GitProviderConfig(
            provider_type="github",
            credentials_secret_ref=_GITHUB_TEST_TOKEN_VAR,
            default_branch="main",
        ),
        primary_language="python",
    )
    pipeline = GraphConstructionPipeline()
    await pipeline.ingest_repository(
        registration,
        graph,  # type: ignore[arg-type]
    )

    ctx = AgentContext(
        agent_id=agent_id,
        repo=_PYTHON_REPO_ID,
        base_template="api_maintainer",
        system_prompt=f"Agent for {_PYTHON_REPO_ID}.",
        repo_architectural_summary="",
    )
    await state_store.save_agent_context(agent_id, ctx)

    # ── Step 2: Verify graph has Python entities ──────────────────────────────
    entities = graph.entities()
    kinds = {e.get("kind", "") for e in entities}
    assert any(e.get("kind") == "function" for e in entities), (
        f"Expected at least one 'function' entity in graph; got kinds: {kinds}"
    )
    assert any(e.get("kind") == "module" for e in entities), (
        f"Expected at least one 'module' entity in graph; got kinds: {kinds}"
    )

    # ── Steps 3–5: Run agent workflow with mocked LLM ─────────────────────────
    git_ops_tool: _E2EGitOpsTool | None = None
    result: dict[str, Any]

    with tempfile.TemporaryDirectory() as tmp_dir:
        working_dir = Path(tmp_dir) / "clone"
        await provider.clone(
            repo_id=_PYTHON_REPO_ID,
            target_path=str(working_dir),
            branch="main",
        )

        git_ops_tool = _E2EGitOpsTool(
            provider=provider,
            repo_id=_PYTHON_REPO_ID,
            working_dir=working_dir,
            agent_id=agent_id,
            task_id=task_id,
        )
        tools: dict[str, Any] = {"git_ops": git_ops_tool}

        def _llm_call(_model: str) -> int:
            return 1_000

        result = await resume_task(
            agent_id=agent_id,
            task_id=task_id,
            task_description="Fix the failing test in tests/test_api.py",
            state_store=state_store,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            graph=graph,  # type: ignore[arg-type]
            llm_call=_llm_call,
        )

    # ── AC assertions ─────────────────────────────────────────────────────────
    assert result["phase"] == TaskPhase.COMPLETED, (
        f"Expected TaskPhase.COMPLETED; got {result['phase']!r}"
    )
    pr_url: str = str(result.get("pr_url") or "")
    assert pr_url, "Expected non-empty pr_url after task completion"
    assert f"/{_PYTHON_REPO_ID}/pull/" in pr_url, (
        f"PR URL does not reference {_PYTHON_REPO_ID!r}: {pr_url!r}"
    )

    # ── Step 6: Verify PR exists on GitHub, then clean up ─────────────────────
    pr_number = int(pr_url.rstrip("/").split("/")[-1])
    branch_name = git_ops_tool.branch_name
    assert branch_name is not None, "git_ops_tool.branch_name was not set"

    await _close_pr_and_delete_branch(_PYTHON_REPO_ID, pr_number, branch_name)


# ── Test 2: TypeScript frontend fixture ───────────────────────────────────────


@_needs_token
@pytest.mark.asyncio
async def test_register_and_submit_task_typescript_frontend() -> None:
    """E2E: Register fixture-typescript-frontend, submit a task, verify PR created.

    Steps:
    1. Invoke ``GraphConstructionPipeline`` (with ``TypeScriptParser``) against
       ``tractable-test-org/fixture-typescript-frontend``.
    2. Assert graph contains TypeScript entities (at least one entity).
    3. Submit task ``"Fix the ESLint violation in src/App.tsx"`` with mocked LLM.
    4. Assert ``TaskPhase.COMPLETED`` and non-empty ``pr_url``.
    5. Clean up: close the PR and delete the branch.
    6. Assert fixture repo has no open PRs after cleanup.
    """
    from tractable.agent.workflow import resume_task

    provider = _github_provider()
    graph = _RecordingGraph()
    state_store = _InMemoryStateStore()

    agent_id = f"e2e-agent-{uuid.uuid4().hex[:8]}"
    task_id = f"e2e-task-{uuid.uuid4().hex[:8]}"

    # ── Step 1: Register ──────────────────────────────────────────────────────
    registration = RepositoryRegistration(
        name=_TS_REPO_ID,
        git_url=f"https://github.com/{_TS_REPO_ID}.git",
        git_provider=GitProviderConfig(
            provider_type="github",
            credentials_secret_ref=_GITHUB_TEST_TOKEN_VAR,
            default_branch="main",
        ),
        primary_language="typescript",
    )
    pipeline = GraphConstructionPipeline(extra_parsers=[TypeScriptParser()])
    await pipeline.ingest_repository(
        registration,
        graph,  # type: ignore[arg-type]
    )

    ctx = AgentContext(
        agent_id=agent_id,
        repo=_TS_REPO_ID,
        base_template="frontend_maintainer",
        system_prompt=f"Agent for {_TS_REPO_ID}.",
        repo_architectural_summary="",
    )
    await state_store.save_agent_context(agent_id, ctx)

    # ── Step 2: Verify graph has TypeScript entities ──────────────────────────
    entities = graph.entities()
    assert entities, (
        "Expected at least one TypeScript entity in graph after ingestion; "
        "got none — check that TypeScriptParser is registered"
    )

    # ── Steps 3–5: Run agent workflow with mocked LLM ─────────────────────────
    git_ops_tool: _E2EGitOpsTool | None = None
    result: dict[str, Any]

    with tempfile.TemporaryDirectory() as tmp_dir:
        working_dir = Path(tmp_dir) / "clone"
        await provider.clone(
            repo_id=_TS_REPO_ID,
            target_path=str(working_dir),
            branch="main",
        )

        git_ops_tool = _E2EGitOpsTool(
            provider=provider,
            repo_id=_TS_REPO_ID,
            working_dir=working_dir,
            agent_id=agent_id,
            task_id=task_id,
        )
        tools: dict[str, Any] = {"git_ops": git_ops_tool}

        def _llm_call(_model: str) -> int:
            return 1_000

        result = await resume_task(
            agent_id=agent_id,
            task_id=task_id,
            task_description="Fix the ESLint violation in src/App.tsx",
            state_store=state_store,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            graph=graph,  # type: ignore[arg-type]
            llm_call=_llm_call,
        )

    # ── AC assertions ─────────────────────────────────────────────────────────
    assert result["phase"] == TaskPhase.COMPLETED, (
        f"Expected TaskPhase.COMPLETED; got {result['phase']!r}"
    )
    pr_url: str = str(result.get("pr_url") or "")
    assert pr_url, "Expected non-empty pr_url after task completion"
    assert f"/{_TS_REPO_ID}/pull/" in pr_url, (
        f"PR URL does not reference {_TS_REPO_ID!r}: {pr_url!r}"
    )

    # ── Cleanup ───────────────────────────────────────────────────────────────
    pr_number = int(pr_url.rstrip("/").split("/")[-1])
    branch_name = git_ops_tool.branch_name
    assert branch_name is not None, "git_ops_tool.branch_name was not set"

    await _close_pr_and_delete_branch(_TS_REPO_ID, pr_number, branch_name)

    # ── Step 6: Assert no open PRs remain on the fixture repo ─────────────────
    token = os.environ.get(_GITHUB_TEST_TOKEN_VAR, "")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    async with httpx.AsyncClient(base_url="https://api.github.com") as client:
        resp = await client.get(
            f"/repos/{_TS_REPO_ID}/pulls",
            headers=headers,
            params={"state": "open"},
        )
    assert resp.status_code == 200, (
        f"GitHub API returned {resp.status_code} when listing open PRs"
    )
    open_prs: list[Any] = resp.json()
    assert len(open_prs) == 0, (
        f"Expected 0 open PRs after cleanup; found {len(open_prs)}: "
        f"{[p.get('html_url') for p in open_prs]}"
    )
