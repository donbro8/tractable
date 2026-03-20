"""Unit tests for tractable/protocols/.

For each Protocol, verifies:
- A minimal stub class that satisfies it passes isinstance()
- A class missing required methods fails isinstance()
- The Protocol has __protocol_attrs__ (i.e. is runtime_checkable)
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timezone
from typing import Any

from tractable.protocols.agent_state_store import AgentStateStore
from tractable.protocols.code_graph import CodeGraph, TemporalCodeGraph
from tractable.protocols.event_bus import AgentEvent, EventBus
from tractable.protocols.git_provider import GitProvider
from tractable.protocols.graph_construction import (
    CodeParser,
    FuzzyResolver,
    ParseResult,
    ResolvedReference,
    UnresolvedReference,
)
from tractable.protocols.reactivity import (
    AgentLifecycleManager,
    ChangeIngestionPipeline,
    ChangePoller,
    RepositoryChangeEvent,
    SyncResult,
    WebhookReceiver,
)
from tractable.protocols.tool import Tool, ToolResult
from tractable.types.agent import AgentCheckpoint, AgentContext, AuditEntry
from tractable.types.enums import ChangeSource, TaskPhase
from tractable.types.git import (
    BranchProtectionRules,
    CheckRunInfo,
    CommitEntry,
    FileEntry,
    MergeResult,
    PullRequestHandle,
)
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
    ChangeNotification,
    ChangeSet,
    GraphDiff,
    TemporalGraphEntity,
    TemporalMutation,
    TemporalMutationResult,
)

NOW = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)


# ── Helpers ────────────────────────────────────────────────────────────


def test_all_protocols_are_runtime_checkable() -> None:
    protocols = [
        GitProvider, CodeGraph, TemporalCodeGraph,
        AgentStateStore, CodeParser, FuzzyResolver,
        EventBus, Tool,
        WebhookReceiver, ChangeIngestionPipeline,
        ChangePoller, AgentLifecycleManager,
    ]
    for proto in protocols:
        assert hasattr(proto, "__protocol_attrs__"), (
            f"{proto.__name__} is not runtime_checkable"
        )


# ── GitProvider ────────────────────────────────────────────────────────


class _GitProviderStub:
    async def clone(self, repo_id: str, target_path: str, branch: str = "main",
                    sparse_paths: Sequence[str] | None = None) -> str: ...
    async def create_branch(self, repo_id: str, branch_name: str,
                            from_ref: str = "main") -> str: ...
    async def create_pull_request(self, repo_id: str, title: str, body: str,
                                  head_branch: str, base_branch: str = "main",
                                  reviewers: Sequence[str] | None = None,
                                  labels: Sequence[str] | None = None) -> PullRequestHandle: ...
    async def merge_pull_request(self, repo_id: str, pr_handle: PullRequestHandle,
                                 strategy: str = "squash") -> MergeResult: ...
    async def get_file_content(self, repo_id: str, file_path: str,
                               ref: str = "main") -> bytes: ...
    async def list_files(self, repo_id: str, path: str = "",
                         ref: str = "main") -> Sequence[FileEntry]: ...
    async def get_diff(self, repo_id: str, base_ref: str, head_ref: str) -> str: ...
    async def get_commit_history(self, repo_id: str, path: str | None = None,
                                 since: datetime | None = None,
                                 limit: int = 50) -> Sequence[CommitEntry]: ...
    async def set_branch_protection(self, repo_id: str, branch: str,
                                    rules: BranchProtectionRules) -> None: ...
    async def get_check_runs(self, repo_id: str, pr_number: int) -> Sequence[CheckRunInfo]: ...
    async def get_check_run_log(self, log_url: str) -> str: ...
    async def rerun_failed_checks(self, repo_id: str, pr_number: int) -> None: ...


def test_git_provider_isinstance() -> None:
    assert isinstance(_GitProviderStub(), GitProvider)


def test_git_provider_missing_method_fails() -> None:
    class _Incomplete:
        async def clone(self, *a: object, **kw: object) -> str: ...

    assert not isinstance(_Incomplete(), GitProvider)


# ── CodeGraph ──────────────────────────────────────────────────────────


class _CodeGraphStub:
    async def query(self, cypher: str, params: dict[str, Any] | None = None) -> Sequence[dict[str, Any]]: ...
    async def get_entity(self, entity_id: str) -> GraphEntity | None: ...
    async def get_neighborhood(self, entity_id: str, depth: int = 2,
                               min_confidence: float = 0.7) -> Subgraph: ...
    async def impact_analysis(self, entity_ids: Sequence[str], depth: int = 3,
                              min_confidence: float = 0.5) -> ImpactReport: ...
    async def get_repo_boundary_edges(self, repo_name: str) -> Sequence[CrossRepoEdge]: ...
    async def get_repo_summary(self, repo_name: str) -> RepoGraphSummary: ...
    async def mutate(self, mutations: Sequence[GraphMutation]) -> MutationResult: ...


def test_code_graph_isinstance() -> None:
    assert isinstance(_CodeGraphStub(), CodeGraph)


# ── TemporalCodeGraph ──────────────────────────────────────────────────


class _TemporalCodeGraphStub:
    async def query_current(self, cypher: str, params: dict[str, Any] | None = None) -> Sequence[dict[str, Any]]: ...
    async def get_current_entity(self, entity_id: str) -> TemporalGraphEntity | None: ...
    async def impact_analysis_current(self, entity_ids: Sequence[str], depth: int = 3, min_confidence: float = 0.5) -> ImpactReport: ...
    async def query_at(self, cypher: str, at_time: datetime, params: dict[str, Any] | None = None) -> Sequence[dict[str, Any]]: ...
    async def get_entity_at(self, entity_id: str, at_time: datetime) -> TemporalGraphEntity | None: ...
    async def get_entity_history(self, entity_id: str, since: datetime | None = None, until: datetime | None = None) -> Sequence[TemporalGraphEntity]: ...
    async def get_changes_since(self, since: datetime, repo: str | None = None, entity_kinds: Sequence[str] | None = None) -> ChangeSet: ...
    async def get_changes_between(self, start: datetime, end: datetime, repo: str | None = None) -> ChangeSet: ...
    async def get_changes_by_commit(self, commit_sha: str) -> ChangeSet: ...
    async def diff_graph(self, time_a: datetime, time_b: datetime, repo: str | None = None) -> GraphDiff: ...
    async def apply_mutations(self, mutations: Sequence[TemporalMutation], change_source: ChangeSource, commit_sha: str | None = None, agent_id: str | None = None) -> TemporalMutationResult: ...


def test_temporal_code_graph_isinstance() -> None:
    assert isinstance(_TemporalCodeGraphStub(), TemporalCodeGraph)


# ── AgentStateStore ────────────────────────────────────────────────────


class _AgentStateStoreStub:
    async def get_agent_context(self, agent_id: str) -> AgentContext: ...
    async def save_agent_context(self, agent_id: str, context: AgentContext) -> None: ...
    async def list_agents(self) -> Sequence[AgentContext]: ...
    async def get_checkpoint(self, agent_id: str, task_id: str) -> AgentCheckpoint | None: ...
    async def save_checkpoint(self, agent_id: str, task_id: str, checkpoint: AgentCheckpoint) -> None: ...
    async def append_audit_entry(self, entry: AuditEntry) -> None: ...
    async def get_audit_log(self, agent_id: str | None = None, task_id: str | None = None,
                            since: datetime | None = None, limit: int = 100) -> Sequence[AuditEntry]: ...


def test_agent_state_store_isinstance() -> None:
    assert isinstance(_AgentStateStoreStub(), AgentStateStore)


# ── CodeParser ─────────────────────────────────────────────────────────


class _CodeParserStub:
    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".py"})

    async def parse_file(self, file_path: str, content: bytes) -> ParseResult: ...


def test_code_parser_isinstance() -> None:
    assert isinstance(_CodeParserStub(), CodeParser)


def test_code_parser_supported_extensions() -> None:
    stub = _CodeParserStub()
    assert ".py" in stub.supported_extensions


# ── FuzzyResolver ──────────────────────────────────────────────────────


class _FuzzyResolverStub:
    async def resolve_batch(self, references: Sequence[UnresolvedReference],
                            candidate_entities: Sequence[GraphEntity]) -> Sequence[ResolvedReference]: ...


def test_fuzzy_resolver_isinstance() -> None:
    assert isinstance(_FuzzyResolverStub(), FuzzyResolver)


# ── EventBus ───────────────────────────────────────────────────────────


class _EventBusStub:
    async def publish(self, topic: str, event: AgentEvent) -> None: ...
    async def subscribe(self, topic: str, agent_id: str) -> AsyncIterator[AgentEvent]: ...


def test_event_bus_isinstance() -> None:
    assert isinstance(_EventBusStub(), EventBus)


# ── Tool ───────────────────────────────────────────────────────────────


class _ToolStub:
    @property
    def name(self) -> str:
        return "test_tool"

    @property
    def description(self) -> str:
        return "A test tool"

    async def invoke(self, params: dict[str, Any]) -> ToolResult: ...


def test_tool_isinstance() -> None:
    assert isinstance(_ToolStub(), Tool)


def test_tool_missing_property_fails() -> None:
    class _Incomplete:
        @property
        def name(self) -> str:
            return "x"
        # missing description and invoke

    assert not isinstance(_Incomplete(), Tool)


# ── WebhookReceiver ────────────────────────────────────────────────────


class _WebhookReceiverStub:
    async def handle_webhook(self, headers: dict[str, str], body: bytes,
                             provider: str) -> RepositoryChangeEvent | None: ...
    async def verify_signature(self, headers: dict[str, str], body: bytes,
                               secret: str) -> bool: ...


def test_webhook_receiver_isinstance() -> None:
    assert isinstance(_WebhookReceiverStub(), WebhookReceiver)


# ── ChangeIngestionPipeline ────────────────────────────────────────────


class _ChangeIngestionPipelineStub:
    async def process_change(self, event: RepositoryChangeEvent) -> Any: ...


def test_change_ingestion_pipeline_isinstance() -> None:
    assert isinstance(_ChangeIngestionPipelineStub(), ChangeIngestionPipeline)


# ── ChangePoller ───────────────────────────────────────────────────────


class _ChangePollerStub:
    async def poll_repo(self, repo_name: str) -> RepositoryChangeEvent | None: ...
    async def poll_all(self) -> Sequence[RepositoryChangeEvent]: ...


def test_change_poller_isinstance() -> None:
    assert isinstance(_ChangePollerStub(), ChangePoller)


# ── AgentLifecycleManager ──────────────────────────────────────────────


class _AgentLifecycleManagerStub:
    async def notify_agent(self, agent_id: str, notification: ChangeNotification) -> None: ...
    async def wake_agent(self, agent_id: str, reason: str) -> None: ...
    async def sync_agent_repo(self, agent_id: str, to_ref: str) -> SyncResult: ...
    async def get_agent_last_active(self, agent_id: str) -> datetime | None: ...


def test_agent_lifecycle_manager_isinstance() -> None:
    assert isinstance(_AgentLifecycleManagerStub(), AgentLifecycleManager)
