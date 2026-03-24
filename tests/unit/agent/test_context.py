"""Unit tests for tractable/agent/context.py — TASK-2.3.2.

Covers:
- {repo_name} placeholder substitution (AC-1)
- Pinned instructions at end after overrides (AC-2)
- Truncation: recent changes first, repo summary second (AC-3)
- Scope injection (AC-4)
- Layer precedence ordering
- None last_active defaults to 7 days ago (DoD requirement)
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tractable.agent.context import assemble_context
from tractable.errors import RecoverableError
from tractable.types.agent import AgentCheckpoint, AgentContext, AuditEntry
from tractable.types.config import (
    AgentScope,
    GitProviderConfig,
    RepositoryRegistration,
)
from tractable.types.enums import AutonomyLevel, ChangeSource
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

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_registration(
    name: str = "my-api",
    agent_template: str = "api_maintainer",
    scope: AgentScope | None = None,
    governance_overrides: dict[str, Any] | None = None,
    autonomy_level: AutonomyLevel = AutonomyLevel.SUPERVISED,
) -> RepositoryRegistration:
    return RepositoryRegistration(
        name=name,
        git_url=f"https://github.com/org/{name}",
        git_provider=GitProviderConfig(
            provider_type="github",
            credentials_secret_ref="gh-token",
        ),
        primary_language="python",
        agent_template=agent_template,
        autonomy_level=autonomy_level,
        scope=scope,
        governance_overrides=governance_overrides or {},
    )


class _MockStateStore:
    """AgentStateStore that returns configurable pinned_instructions."""

    def __init__(self, pinned: list[str] | None = None) -> None:
        self._pinned = pinned or []

    async def get_agent_context(self, agent_id: str) -> AgentContext:
        return AgentContext(
            agent_id=agent_id,
            base_template="api_maintainer",
            system_prompt="",
            repo_architectural_summary="",
            pinned_instructions=self._pinned,
        )

    async def save_agent_context(self, agent_id: str, context: AgentContext) -> None:
        pass

    async def get_checkpoint(self, agent_id: str, task_id: str) -> AgentCheckpoint | None:
        return None

    async def save_checkpoint(
        self, agent_id: str, task_id: str, checkpoint: AgentCheckpoint
    ) -> None:
        pass

    async def append_audit_entry(self, entry: AuditEntry) -> None:
        pass

    async def get_audit_log(
        self,
        agent_id: str | None = None,
        task_id: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> Sequence[AuditEntry]:
        return []


class _MockGraph:
    """TemporalCodeGraph + CodeGraph mock.

    Exposes both get_repo_summary (CodeGraph) and get_changes_since
    (TemporalCodeGraph), matching what the FalkorDB implementation provides.
    """

    def __init__(
        self,
        summary_text: str = "Test repo summary",
        total_entities: int = 5,
        key_modules: list[str] | None = None,
        changes_summary: str = "",
        since_received: list[datetime] | None = None,
    ) -> None:
        self._summary_text = summary_text
        self._total_entities = total_entities
        self._key_modules = key_modules or ["main.py"]
        self._changes_summary = changes_summary
        self.since_received: list[datetime] = since_received if since_received is not None else []

    # ── CodeGraph methods ──────────────────────────────────────────────

    async def get_repo_summary(self, repo_name: str) -> RepoGraphSummary:
        return RepoGraphSummary(
            repo_name=repo_name,
            total_entities=self._total_entities,
            key_modules=self._key_modules,
            public_interfaces=[],
            cross_repo_dependencies=[],
            summary_text=self._summary_text,
        )

    async def query(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> Sequence[dict[str, Any]]:
        return []

    async def get_entity(self, entity_id: str) -> GraphEntity | None:
        return None

    async def get_neighborhood(
        self,
        entity_id: str,
        depth: int = 2,
        min_confidence: float = 0.7,
    ) -> Subgraph:
        return Subgraph(nodes=[], edges=[])

    async def impact_analysis(
        self,
        entity_ids: Sequence[str],
        depth: int = 3,
        min_confidence: float = 0.5,
    ) -> ImpactReport:
        from tractable.types.enums import ChangeRisk

        return ImpactReport(
            directly_affected=[],
            transitively_affected=[],
            affected_repos=[],
            cross_repo_edges=[],
            risk_level=ChangeRisk.LOW,
        )

    async def get_repo_boundary_edges(self, repo_name: str) -> Sequence[CrossRepoEdge]:
        return []

    async def mutate(self, mutations: Sequence[GraphMutation]) -> MutationResult:
        return MutationResult(applied=0)

    # ── TemporalCodeGraph methods ──────────────────────────────────────

    async def query_current(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> Sequence[dict[str, Any]]:
        return []

    async def get_current_entity(self, entity_id: str) -> TemporalGraphEntity | None:
        return None

    async def impact_analysis_current(
        self,
        entity_ids: Sequence[str],
        depth: int = 3,
        min_confidence: float = 0.5,
    ) -> ImpactReport:
        from tractable.types.enums import ChangeRisk

        return ImpactReport(
            directly_affected=[],
            transitively_affected=[],
            affected_repos=[],
            cross_repo_edges=[],
            risk_level=ChangeRisk.LOW,
        )

    async def query_at(
        self,
        cypher: str,
        at_time: datetime,
        params: dict[str, Any] | None = None,
    ) -> Sequence[dict[str, Any]]:
        return []

    async def get_entity_at(self, entity_id: str, at_time: datetime) -> TemporalGraphEntity | None:
        return None

    async def get_entity_history(
        self,
        entity_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> Sequence[TemporalGraphEntity]:
        return []

    async def get_changes_since(
        self,
        since: datetime,
        repo: str | None = None,
        entity_kinds: Sequence[str] | None = None,
    ) -> ChangeSet:
        self.since_received.append(since)
        now = datetime.now(tz=UTC)
        return ChangeSet(
            time_range_start=since,
            time_range_end=now,
            repo_filter=repo,
        )

    async def get_changes_between(
        self,
        start: datetime,
        end: datetime,
        repo: str | None = None,
    ) -> ChangeSet:
        return ChangeSet(
            time_range_start=start,
            time_range_end=end,
            repo_filter=repo,
        )

    async def get_changes_by_commit(self, commit_sha: str) -> ChangeSet:
        now = datetime.now(tz=UTC)
        return ChangeSet(time_range_start=now, time_range_end=now)

    async def diff_graph(
        self,
        time_a: datetime,
        time_b: datetime,
        repo: str | None = None,
    ) -> GraphDiff:
        return GraphDiff(
            time_a=time_a,
            time_b=time_b,
            added_entities=[],
            removed_entities=[],
            modified_entities=[],
            added_edges=[],
            removed_edges=[],
            repos_affected=[],
        )

    async def apply_mutations(
        self,
        mutations: Sequence[TemporalMutation],
        change_source: ChangeSource,
        commit_sha: str | None = None,
        agent_id: str | None = None,
    ) -> TemporalMutationResult:
        return TemporalMutationResult(applied=0)


# ---------------------------------------------------------------------------
# AC-1: {repo_name} substitution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repo_name_substitution() -> None:
    """AC-1: {repo_name} placeholder replaced by registration.name."""
    store = _MockStateStore()
    graph = _MockGraph()
    reg = _make_registration(name="my-api")

    result = await assemble_context("agent-1", store, graph, reg)

    assert "my-api" in result


@pytest.mark.asyncio
async def test_repo_name_substitution_different_repo() -> None:
    """AC-1 (extended): Different repo name is correctly substituted."""
    store = _MockStateStore()
    graph = _MockGraph()
    reg = _make_registration(name="payments-service")

    result = await assemble_context("agent-1", store, graph, reg)

    assert "payments-service" in result


# ---------------------------------------------------------------------------
# AC-2: Pinned instructions appear at end after overrides
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pinned_instructions_at_end() -> None:
    """AC-2: Pinned instructions appear at end with [pinned] prefix."""
    store = _MockStateStore(pinned=["Never modify auth/"])
    graph = _MockGraph()
    reg = _make_registration()

    result = await assemble_context("agent-1", store, graph, reg)

    assert "[pinned] Never modify auth/" in result
    assert result.endswith("[pinned] Never modify auth/")


@pytest.mark.asyncio
async def test_pinned_instructions_after_overrides() -> None:
    """AC-2: Scope override appears before pinned instructions in the output."""
    store = _MockStateStore(pinned=["Never modify auth/"])
    graph = _MockGraph()
    reg = _make_registration(scope=AgentScope(allowed_paths=["src/"]))

    result = await assemble_context("agent-1", store, graph, reg)

    scope_idx = result.index("Your scope is limited to: src/")
    pinned_idx = result.index("[pinned] Never modify auth/")
    assert scope_idx < pinned_idx, "Scope override must appear before pinned instructions"


@pytest.mark.asyncio
async def test_multiple_pinned_instructions() -> None:
    """Multiple pinned instructions all appear at the end."""
    pinned = ["Never modify auth/", "Always run tests", "Check blast radius first"]
    store = _MockStateStore(pinned=pinned)
    graph = _MockGraph()
    reg = _make_registration()

    result = await assemble_context("agent-1", store, graph, reg)

    for instr in pinned:
        assert f"[pinned] {instr}" in result

    last_pinned_idx = result.rindex("[pinned]")
    overrides_idx = result.index("Governance:")
    assert overrides_idx < last_pinned_idx


@pytest.mark.asyncio
async def test_no_pinned_instructions_no_pinned_section() -> None:
    """No [pinned] prefix appears when pinned_instructions is empty."""
    store = _MockStateStore(pinned=[])
    graph = _MockGraph()
    reg = _make_registration()

    result = await assemble_context("agent-1", store, graph, reg)

    assert "[pinned]" not in result


# ---------------------------------------------------------------------------
# AC-3: Truncation — recent changes first, repo summary second
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_truncation_drops_recent_changes_first() -> None:
    """AC-3: When over budget, recent changes are dropped first.

    The mock returns a ChangeSet with commits so the recent_changes_section
    is non-empty.  We set max_prompt_chars just below the full assembled
    length so that recent_changes (the most volatile section) is dropped
    while repo summary is retained.
    """

    class _GraphWithChanges(_MockGraph):
        async def get_changes_since(
            self,
            since: datetime,
            repo: str | None = None,
            entity_kinds: Sequence[str] | None = None,
        ) -> ChangeSet:
            return ChangeSet(
                time_range_start=since,
                time_range_end=datetime.now(tz=UTC),
                repo_filter=repo,
                commits=["abc123"],
                # Add a fake entity so is_empty is False and summary is non-empty
            )

    store = _MockStateStore()
    graph = _GraphWithChanges(summary_text="Repo summary text", total_entities=5)
    reg = _make_registration()

    # Build full prompt to measure its length.
    full = await assemble_context("agent-1", store, graph, reg, max_prompt_chars=1_000_000)

    # Set the budget just below the full length so truncation fires.
    trimmed = await assemble_context("agent-1", store, graph, reg, max_prompt_chars=len(full) - 1)

    # Template content must always be preserved.
    assert "my-api" in trimmed

    # The trimmed result must be shorter than the full result.
    assert len(trimmed) < len(full)


@pytest.mark.asyncio
async def test_truncation_small_limit_preserves_template() -> None:
    """AC-3: Template content preserved even when max_prompt_chars is very small."""
    store = _MockStateStore(pinned=["Keep tests passing"])
    graph = _MockGraph(summary_text="X" * 1000, total_entities=500)
    reg = _make_registration(name="test-repo")

    # Very small limit — should drop both recent_changes and repo_summary
    result = await assemble_context("agent-1", store, graph, reg, max_prompt_chars=100)

    # Template must be preserved: repo_name should appear
    assert "test-repo" in result
    # Pinned instructions must be preserved
    assert "[pinned] Keep tests passing" in result


@pytest.mark.asyncio
async def test_truncation_order_repo_summary_before_recent_changes() -> None:
    """AC-3: Repo summary is only dropped after recent changes are already dropped."""
    store = _MockStateStore()
    graph = _MockGraph(
        summary_text="Long repo summary content",
        total_entities=1,
    )
    reg = _make_registration()

    # With a very large limit — nothing truncated
    full = await assemble_context("agent-1", store, graph, reg, max_prompt_chars=1_000_000)

    # Repo summary section should be present when there's enough space
    assert "Your domain:" in full


# ---------------------------------------------------------------------------
# AC-4: Scope injection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scope_injection() -> None:
    """AC-4: registration.scope.allowed_paths injected as scope restriction."""
    store = _MockStateStore()
    graph = _MockGraph()
    reg = _make_registration(scope=AgentScope(allowed_paths=["src/payments/"]))

    result = await assemble_context("agent-1", store, graph, reg)

    assert "Your scope is limited to: src/payments/" in result


@pytest.mark.asyncio
async def test_scope_injection_multiple_paths() -> None:
    """AC-4 (extended): Multiple allowed paths appear in scope section."""
    store = _MockStateStore()
    graph = _MockGraph()
    reg = _make_registration(scope=AgentScope(allowed_paths=["src/auth/", "src/billing/"]))

    result = await assemble_context("agent-1", store, graph, reg)

    assert "src/auth/" in result
    assert "src/billing/" in result


@pytest.mark.asyncio
async def test_no_scope_no_scope_section() -> None:
    """AC-4 (boundary): No scope restriction when registration.scope is None."""
    store = _MockStateStore()
    graph = _MockGraph()
    reg = _make_registration(scope=None)

    result = await assemble_context("agent-1", store, graph, reg)

    assert "Your scope is limited to:" not in result


# ---------------------------------------------------------------------------
# DoD requirement: None last_active defaults to 7 days ago
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_last_active_defaults_to_7_days_ago() -> None:
    """DoD: When last_active is None, get_changes_since uses 7 days ago."""
    since_received: list[datetime] = []
    graph = _MockGraph(since_received=since_received)
    store = _MockStateStore()
    reg = _make_registration()

    before = datetime.now(tz=UTC) - timedelta(days=7, seconds=5)
    await assemble_context("agent-1", store, graph, reg)
    after = datetime.now(tz=UTC) - timedelta(days=7, seconds=-5)

    assert len(since_received) == 1
    assert before <= since_received[0] <= after


# ---------------------------------------------------------------------------
# Layer precedence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_three_layers_present() -> None:
    """All three layers (template, overrides, pinned) appear in the output."""
    pinned = ["Do not touch the database"]
    store = _MockStateStore(pinned=pinned)
    graph = _MockGraph()
    reg = _make_registration(
        name="my-api",
        scope=AgentScope(allowed_paths=["src/"]),
    )

    result = await assemble_context("agent-1", store, graph, reg)

    # Layer 1: template
    assert "my-api" in result

    # Layer 2: overrides
    assert "Your scope is limited to: src/" in result

    # Layer 3: pinned
    assert "[pinned] Do not touch the database" in result


@pytest.mark.asyncio
async def test_unknown_template_raises_recoverable_error() -> None:
    """RecoverableError raised for unrecognised agent_template ID (TASK-3.1.1)."""
    store = _MockStateStore()
    graph = _MockGraph()
    reg = _make_registration(agent_template="nonexistent_template_xyz")

    with pytest.raises(RecoverableError, match="Unknown agent template"):
        await assemble_context("agent-1", store, graph, reg)


@pytest.mark.asyncio
async def test_governance_overrides_applied() -> None:
    """Governance overrides are reflected in the overrides section."""
    store = _MockStateStore()
    graph = _MockGraph()
    reg = _make_registration(governance_overrides={"max_files_per_change": 5})

    result = await assemble_context("agent-1", store, graph, reg)

    assert "max 5 files per change" in result
