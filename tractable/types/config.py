"""Configuration and registration models for the Tractable framework.

All configuration is Python, not YAML. Teams onboard a repository by
instantiating ``RepositoryRegistration`` and submitting it. Validation is
automatic via Pydantic.

Sources:
- tech-spec.py §3 — Configuration Models
- realtime-temporal-spec.py §D — AgentReactivityConfig
- plan.md — AgentScope definition
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, Field

from tractable.types.enums import AgentStatus, AutonomyLevel, ChangeRisk

# ── Capability ─────────────────────────────────────────────────────────


class Capability(BaseModel):
    """A single thing an agent is allowed to do."""

    name: str
    description: str
    risk_level: ChangeRisk = ChangeRisk.LOW
    requires_approval: bool = False
    approval_from: Literal["human", "coordinator", "any_agent"] | None = None


# Pre-defined capability constants — compose these, don't inherit.
CODE_READ = Capability(name="code_read", description="Read files from assigned repository")
CODE_WRITE = Capability(name="code_write", description="Write/modify files in assigned repository")
TEST_RUN = Capability(name="test_run", description="Execute test suites")
LINT_RUN = Capability(name="lint_run", description="Execute linters and formatters")
GRAPH_QUERY_SCOPED = Capability(
    name="graph_query_scoped",
    description="Query code graph within assigned repo + 1-hop neighbors",
)
GRAPH_QUERY_GLOBAL = Capability(
    name="graph_query_global",
    description="Query entire code graph across all repos",
)
PR_CREATE = Capability(name="pr_create", description="Create pull requests")
PR_MERGE = Capability(
    name="pr_merge",
    description="Merge pull requests",
    risk_level=ChangeRisk.MEDIUM,
    requires_approval=True,
    approval_from="human",
)
BRANCH_CREATE = Capability(name="branch_create", description="Create git branches")
DEPENDENCY_UPDATE = Capability(
    name="dependency_update",
    description="Update package dependencies",
    risk_level=ChangeRisk.MEDIUM,
)
SPAWN_SUBAGENT = Capability(
    name="spawn_subagent",
    description="Create child agents for subtask delegation",
)
MESSAGE_AGENTS = Capability(
    name="message_agents",
    description="Send coordination messages to other agents",
)
TERRAFORM_PLAN = Capability(name="terraform_plan", description="Run terraform plan")
TERRAFORM_APPLY = Capability(
    name="terraform_apply",
    description="Run terraform apply",
    risk_level=ChangeRisk.CRITICAL,
    requires_approval=True,
    approval_from="human",
)
CDK_SYNTH = Capability(name="cdk_synth", description="Run cdk synth to generate CloudFormation")
CDK_DEPLOY = Capability(
    name="cdk_deploy",
    description="Run cdk deploy",
    risk_level=ChangeRisk.CRITICAL,
    requires_approval=True,
    approval_from="human",
)
K8S_APPLY = Capability(
    name="k8s_apply",
    description="Apply Kubernetes manifests",
    risk_level=ChangeRisk.HIGH,
    requires_approval=True,
    approval_from="human",
)
DB_MIGRATE = Capability(
    name="db_migrate",
    description="Run database migrations",
    risk_level=ChangeRisk.CRITICAL,
    requires_approval=True,
    approval_from="human",
)


# ── Governance ─────────────────────────────────────────────────────────


class SensitivePathRule(BaseModel):
    """A path glob that requires extra governance before modification."""

    pattern: str
    reason: str
    policy: Literal[
        "human_review_always",
        "human_approval_required",
        "coordinator_review",
        "block",
    ]


class CrossRepoChangePolicy(BaseModel):
    """Policy for changes that affect multiple repositories."""

    require_impact_analysis: bool = True
    require_affected_agent_approval: bool = True
    min_confidence_for_blocking: float = 0.7
    notification_confidence_threshold: float = 0.3


class GovernancePolicy(BaseModel):
    """Rules constraining how an agent operates."""

    max_files_per_change: int = 20
    max_lines_per_change: int = 500
    requires_tests_pass: bool = True
    requires_lint_pass: bool = True
    require_pr_for_changes: bool = True
    auto_merge_allowed: bool = False
    max_retries_on_failure: int = 3
    token_budget_per_task: int = 200_000
    token_budget_per_day: int = 2_000_000
    escalation_on_failure: str = "system_manager"
    sensitive_path_patterns: list[SensitivePathRule] = []
    cross_repo_change_policy: CrossRepoChangePolicy | None = None


# ── Context configuration ──────────────────────────────────────────────


class ContextConfig(BaseModel):
    """How the agent manages its context window."""

    max_context_tokens: int = 128_000
    compaction_threshold: float = 0.8
    compaction_strategy: Literal[
        "summarize", "drop_old_tool_outputs", "progressive"
    ] = "progressive"
    graph_summary_refresh_interval_minutes: int = 60
    include_cross_repo_digest: bool = True
    max_files_in_working_set: int = 15


# ── Agent templates ────────────────────────────────────────────────────


class AgentTemplate(BaseModel):
    """
    A reusable template for a category of agent.

    Templates are NOT base classes. An agent is configured by selecting
    a template and optionally overriding specific fields or composing
    additional capabilities.
    """

    template_id: str
    description: str
    system_prompt_template: str
    capabilities: list[Capability]
    governance: GovernancePolicy
    context_config: ContextConfig
    recommended_model: str = "claude-sonnet-4"
    escalation_model: str = "claude-opus-4"
    tools_required: list[str] = Field(default_factory=list)


API_MAINTAINER_TEMPLATE = AgentTemplate(
    template_id="api_maintainer",
    description=(
        "Maintains an API service repository. Understands REST/GraphQL conventions, "
        "endpoint design, request validation, and API versioning."
    ),
    system_prompt_template="""\
You are the maintainer of the {repo_name} API repository.

Your domain: {repo_architectural_summary}

Core responsibilities:
- Implement feature requests and bug fixes within this API
- Maintain API contracts (OpenAPI specs, GraphQL schemas)
- Ensure backward compatibility — breaking changes require coordination
- Write and maintain tests for all endpoints
- Keep dependencies up to date

Cross-system awareness:
{cross_repo_digest}

{pinned_instructions}
""",
    capabilities=[
        CODE_READ, CODE_WRITE, TEST_RUN, LINT_RUN,
        GRAPH_QUERY_SCOPED, PR_CREATE, BRANCH_CREATE,
        DEPENDENCY_UPDATE, MESSAGE_AGENTS,
    ],
    governance=GovernancePolicy(
        requires_tests_pass=True,
        requires_lint_pass=True,
        sensitive_path_patterns=[
            SensitivePathRule(
                pattern="**/openapi/**",
                reason="API contract changes affect consumers",
                policy="coordinator_review",
            ),
        ],
        cross_repo_change_policy=CrossRepoChangePolicy(),
    ),
    context_config=ContextConfig(),
    tools_required=[
        "code_editor", "test_runner", "linter",
        "graph_query_mcp", "git_operations", "lsp_server",
    ],
)

INFRA_MAINTAINER_TEMPLATE = AgentTemplate(
    template_id="infra_maintainer",
    description=(
        "Maintains infrastructure-as-code repositories. Works regardless of whether "
        "the repo uses Terraform, CDK, Pulumi, CloudFormation, or Kubernetes manifests."
    ),
    system_prompt_template="""\
You are the infrastructure maintainer for {repo_name}.

Your domain: {repo_architectural_summary}

Core responsibilities:
- Manage infrastructure definitions safely and reproducibly
- NEVER apply destructive changes without human approval
- Always run plan/synth/diff BEFORE any apply/deploy
- Understand the blast radius of every change via graph queries
- Maintain separation of concerns between environments

Safety rules (NON-NEGOTIABLE):
- Run plan/synth first, review output, THEN propose apply
- Any resource deletion MUST be flagged for human review
- IAM, security group, and network changes require human approval
- Always check graph for services depending on changed resources

Cross-system awareness:
{cross_repo_digest}

{pinned_instructions}
""",
    capabilities=[
        CODE_READ, CODE_WRITE, TEST_RUN, LINT_RUN,
        GRAPH_QUERY_SCOPED, PR_CREATE, BRANCH_CREATE,
        MESSAGE_AGENTS,
    ],
    governance=GovernancePolicy(
        requires_tests_pass=True,
        requires_lint_pass=True,
        auto_merge_allowed=False,
        sensitive_path_patterns=[
            SensitivePathRule(
                pattern="**/*.tf",
                reason="Infrastructure definitions",
                policy="human_review_always",
            ),
            SensitivePathRule(
                pattern="**/*.tfvars",
                reason="Infrastructure variables",
                policy="human_review_always",
            ),
            SensitivePathRule(
                pattern="**/cdk.out/**",
                reason="Synthesized templates",
                policy="human_review_always",
            ),
            SensitivePathRule(
                pattern="**/manifests/**",
                reason="Kubernetes manifests",
                policy="human_review_always",
            ),
        ],
        cross_repo_change_policy=CrossRepoChangePolicy(
            require_impact_analysis=True,
            require_affected_agent_approval=True,
        ),
    ),
    context_config=ContextConfig(include_cross_repo_digest=True),
    tools_required=["code_editor", "test_runner", "graph_query_mcp", "git_operations"],
)

FRONTEND_MAINTAINER_TEMPLATE = AgentTemplate(
    template_id="frontend_maintainer",
    description="Maintains frontend application repositories (React, Vue, Angular, etc.).",
    system_prompt_template="""\
You are the frontend maintainer for {repo_name}.

Your domain: {repo_architectural_summary}

Core responsibilities:
- Implement UI features and fix frontend bugs
- Maintain component library consistency
- Ensure accessibility standards
- Keep API client code in sync with backend contracts
- Write and maintain frontend tests

Cross-system awareness (API contracts you consume):
{cross_repo_digest}

{pinned_instructions}
""",
    capabilities=[
        CODE_READ, CODE_WRITE, TEST_RUN, LINT_RUN,
        GRAPH_QUERY_SCOPED, PR_CREATE, BRANCH_CREATE,
        DEPENDENCY_UPDATE, MESSAGE_AGENTS,
    ],
    governance=GovernancePolicy(requires_tests_pass=True, requires_lint_pass=True),
    context_config=ContextConfig(),
    tools_required=[
        "code_editor", "test_runner", "linter",
        "graph_query_mcp", "git_operations", "build_runner",
    ],
)

SHARED_LIB_MAINTAINER_TEMPLATE = AgentTemplate(
    template_id="shared_lib_maintainer",
    description="Maintains shared library repositories consumed by multiple other repos.",
    system_prompt_template="""\
You are the maintainer of {repo_name}, a shared library.

Your domain: {repo_architectural_summary}

Core responsibilities:
- Maintain the library's public API surface
- ANY change to exported interfaces MUST trigger impact analysis
- Ensure semantic versioning is respected
- Breaking changes require coordination with ALL consumers
- Maintain comprehensive tests and documentation

Consumers of this library:
{cross_repo_digest}

{pinned_instructions}
""",
    capabilities=[
        CODE_READ, CODE_WRITE, TEST_RUN, LINT_RUN,
        GRAPH_QUERY_GLOBAL, PR_CREATE, BRANCH_CREATE, MESSAGE_AGENTS,
    ],
    governance=GovernancePolicy(
        requires_tests_pass=True,
        requires_lint_pass=True,
        sensitive_path_patterns=[
            SensitivePathRule(
                pattern="**/public/**",
                reason="Public API surface — consumers depend on this",
                policy="coordinator_review",
            ),
        ],
        cross_repo_change_policy=CrossRepoChangePolicy(
            require_impact_analysis=True,
            require_affected_agent_approval=True,
            min_confidence_for_blocking=0.5,
        ),
    ),
    context_config=ContextConfig(include_cross_repo_digest=True),
    tools_required=[
        "code_editor", "test_runner", "linter",
        "graph_query_mcp", "git_operations", "lsp_server",
    ],
)

COORDINATOR_TEMPLATE = AgentTemplate(
    template_id="coordinator",
    description=(
        "Coordinates changes that span multiple repositories. "
        "Does NOT write code — delegates to repo agents."
    ),
    system_prompt_template="""\
You are a cross-repository coordinator managing: {assigned_repos}.

Your role: Decompose multi-repo tasks into ordered sub-tasks,
assign them to the appropriate repo agents, manage dependencies
between phases, and ensure the overall change is consistent.

You do NOT write code. You:
1. Analyze the task's cross-repo impact via graph queries
2. Create a phased execution plan respecting dependency order
3. Delegate phases to repo agents
4. Monitor progress and handle failures
5. Verify integration after all phases complete

Current system state:
{cross_repo_digest}

{pinned_instructions}
""",
    capabilities=[GRAPH_QUERY_GLOBAL, SPAWN_SUBAGENT, MESSAGE_AGENTS],
    governance=GovernancePolicy(
        auto_merge_allowed=False,
        max_files_per_change=0,
    ),
    context_config=ContextConfig(
        include_cross_repo_digest=True,
        max_context_tokens=200_000,
    ),
    recommended_model="claude-opus-4",
    tools_required=["graph_query_mcp", "event_bus"],
)

TEMPLATE_REGISTRY: dict[str, AgentTemplate] = {
    t.template_id: t
    for t in [
        API_MAINTAINER_TEMPLATE,
        INFRA_MAINTAINER_TEMPLATE,
        FRONTEND_MAINTAINER_TEMPLATE,
        SHARED_LIB_MAINTAINER_TEMPLATE,
        COORDINATOR_TEMPLATE,
    ]
}


# ── Repository registration ────────────────────────────────────────────


class AgentScope(BaseModel):
    """
    Optional path-based scope for an agent within a repository.

    ``deny_paths`` takes precedence over ``allowed_paths``. Enforcement
    is the responsibility of the ``code_editor`` tool at write time, not
    this model.
    """

    allowed_paths: list[str] = Field(default_factory=list)
    allowed_extensions: list[str] = Field(default_factory=list)
    deny_paths: list[str] = Field(default_factory=list)


class DependencyDeclaration(BaseModel):
    """A declared relationship between this repo and another."""

    target_repo: str
    direction: Literal["consumes", "provides"]
    interface_type: Literal[
        "library_import",
        "rest_api",
        "grpc",
        "graphql",
        "event_stream",
        "shared_database",
        "file_system",
        "config_reference",
    ]
    spec_path: str | None = None
    description: str = ""


class GitProviderConfig(BaseModel):
    """Connection details for a git hosting provider."""

    provider_type: Literal["github", "gitlab", "codecommit", "bitbucket", "custom"]
    base_url: str | None = None
    credentials_secret_ref: str
    default_branch: str = "main"


class RepositoryRegistration(BaseModel):
    """
    Entry point for onboarding a repository into the system.
    Teams write a short Python file that instantiates this model.
    """

    name: str
    git_url: str
    git_provider: GitProviderConfig

    primary_language: str
    secondary_languages: list[str] = Field(default_factory=list)
    monorepo_paths: list[str] | None = None

    dependencies: list[DependencyDeclaration] = []

    agent_template: str = "api_maintainer"
    autonomy_level: AutonomyLevel = AutonomyLevel.SUPERVISED
    additional_capabilities: list[Capability] = []
    capability_overrides: list[str] = Field(default_factory=list)
    governance_overrides: dict[str, Any] = Field(default_factory=dict)
    context_overrides: dict[str, Any] = Field(default_factory=dict)

    pinned_instructions: list[str] = Field(default_factory=list)

    webhook_secret: str | None = None
    poll_interval_seconds: int = 60

    scope: AgentScope | None = None

    custom_parsers: list[str] = Field(default_factory=list)
    ignore_patterns: list[str] = Field(
        default_factory=lambda: [
            "node_modules/**",
            ".git/**",
            "__pycache__/**",
            "*.pyc",
            ".terraform/**",
            "cdk.out/**",
            "dist/**",
            "build/**",
            ".next/**",
        ]
    )


# ── Agent instance configuration ──────────────────────────────────────


class AgentInstanceConfig(BaseModel):
    """
    Fully-resolved configuration for a running agent.

    Created by composing a template with per-repository registration
    overrides. This is what the agent runtime reads.
    """

    agent_id: str
    assigned_repo: str
    template_id: str
    model: str
    escalation_model: str
    autonomy_level: AutonomyLevel
    capabilities: list[Capability]
    governance: GovernancePolicy
    context_config: ContextConfig
    tools: list[str]
    git_provider: GitProviderConfig
    status: AgentStatus = AgentStatus.DORMANT
    created_at: datetime | None = None

    @classmethod
    def from_registration(
        cls,
        registration: RepositoryRegistration,
        agent_id: str,
    ) -> Self:
        """Compose a resolved agent config from a template + registration overrides."""
        template = TEMPLATE_REGISTRY[registration.agent_template]

        caps = list(template.capabilities)
        caps.extend(registration.additional_capabilities)
        caps = [c for c in caps if c.name not in registration.capability_overrides]

        gov_dict: dict[str, Any] = template.governance.model_dump()
        gov_dict.update(registration.governance_overrides)
        governance = GovernancePolicy.model_validate(gov_dict)

        ctx_dict: dict[str, Any] = template.context_config.model_dump()
        ctx_dict.update(registration.context_overrides)
        context_config = ContextConfig.model_validate(ctx_dict)

        return cls(
            agent_id=agent_id,
            assigned_repo=registration.name,
            template_id=registration.agent_template,
            model=template.recommended_model,
            escalation_model=template.escalation_model,
            autonomy_level=registration.autonomy_level,
            capabilities=caps,
            governance=governance,
            context_config=context_config,
            tools=list(template.tools_required),
            git_provider=registration.git_provider,
        )


# ── Reactivity configuration ───────────────────────────────────────────


class AgentReactivityConfig(BaseModel):
    """
    Controls how an agent responds to real-time change events.
    Set per-registration alongside the governance policy.
    """

    wake_on_direct_change: bool = True
    wake_on_dependency_change: bool = True
    wake_on_consumer_change: bool = False
    wake_on_transitive_change: bool = False

    catchup_strategy: Literal["full_diff", "summary_only", "entity_focused"] = "full_diff"

    debounce_seconds: int = 30
    batch_changes: bool = True

    auto_pull_on_change: bool = True
    rebase_strategy: Literal["rebase", "merge", "abort"] = "rebase"
