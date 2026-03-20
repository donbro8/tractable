"""Unit tests for tractable/types/config.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tractable.types.config import (
    # Capability constants
    CODE_READ,
    CODE_WRITE,
    DB_MIGRATE,
    PR_MERGE,
    TEMPLATE_REGISTRY,
    TERRAFORM_APPLY,
    AgentInstanceConfig,
    AgentReactivityConfig,
    AgentScope,
    Capability,
    ContextConfig,
    GitProviderConfig,
    GovernancePolicy,
    RepositoryRegistration,
    SensitivePathRule,
)
from tractable.types.enums import AgentStatus, AutonomyLevel, ChangeRisk

# ── Capability ─────────────────────────────────────────────────────────


def test_capability_defaults() -> None:
    cap = Capability(name="custom", description="A custom capability")
    assert cap.risk_level is ChangeRisk.LOW
    assert cap.requires_approval is False
    assert cap.approval_from is None


def test_capability_constants_exist() -> None:
    assert CODE_READ.name == "code_read"
    assert CODE_WRITE.name == "code_write"
    assert PR_MERGE.requires_approval is True
    assert PR_MERGE.approval_from == "human"
    assert TERRAFORM_APPLY.risk_level is ChangeRisk.CRITICAL
    assert DB_MIGRATE.risk_level is ChangeRisk.CRITICAL


def test_all_18_capability_constants_importable() -> None:
    from tractable.types.config import (
        BRANCH_CREATE,
        CDK_DEPLOY,
        CDK_SYNTH,
        CODE_READ,
        CODE_WRITE,
        DB_MIGRATE,
        DEPENDENCY_UPDATE,
        GRAPH_QUERY_GLOBAL,
        GRAPH_QUERY_SCOPED,
        K8S_APPLY,
        LINT_RUN,
        MESSAGE_AGENTS,
        PR_CREATE,
        PR_MERGE,
        SPAWN_SUBAGENT,
        TERRAFORM_APPLY,
        TERRAFORM_PLAN,
        TEST_RUN,
    )

    constants = [
        CODE_READ,
        CODE_WRITE,
        TEST_RUN,
        LINT_RUN,
        GRAPH_QUERY_SCOPED,
        GRAPH_QUERY_GLOBAL,
        PR_CREATE,
        PR_MERGE,
        BRANCH_CREATE,
        DEPENDENCY_UPDATE,
        SPAWN_SUBAGENT,
        MESSAGE_AGENTS,
        TERRAFORM_PLAN,
        TERRAFORM_APPLY,
        CDK_SYNTH,
        CDK_DEPLOY,
        K8S_APPLY,
        DB_MIGRATE,
    ]
    assert len(constants) == 18
    names = {c.name for c in constants}
    assert len(names) == 18  # all unique


# ── GovernancePolicy ───────────────────────────────────────────────────


def test_governance_policy_defaults() -> None:
    g = GovernancePolicy()
    assert g.max_files_per_change == 20
    assert g.requires_tests_pass is True
    assert g.auto_merge_allowed is False
    assert g.sensitive_path_patterns == []
    assert g.cross_repo_change_policy is None


def test_governance_policy_sensitive_paths() -> None:
    rule = SensitivePathRule(
        pattern="migrations/**",
        reason="DB schema changes",
        policy="human_approval_required",
    )
    g = GovernancePolicy(sensitive_path_patterns=[rule])
    assert len(g.sensitive_path_patterns) == 1
    assert g.sensitive_path_patterns[0].policy == "human_approval_required"


def test_sensitive_path_rule_invalid_policy() -> None:
    with pytest.raises(ValidationError):
        SensitivePathRule(pattern="**", reason="test", policy="yolo")  # type: ignore[arg-type]


# ── ContextConfig ──────────────────────────────────────────────────────


def test_context_config_defaults() -> None:
    c = ContextConfig()
    assert c.max_context_tokens == 128_000
    assert c.compaction_strategy == "progressive"
    assert c.include_cross_repo_digest is True


# ── AgentTemplate / TEMPLATE_REGISTRY ─────────────────────────────────


def test_template_registry_has_5_templates() -> None:
    assert len(TEMPLATE_REGISTRY) == 5
    expected = {
        "api_maintainer",
        "infra_maintainer",
        "frontend_maintainer",
        "shared_lib_maintainer",
        "coordinator",
    }
    assert set(TEMPLATE_REGISTRY.keys()) == expected


def test_api_maintainer_template_capabilities() -> None:
    t = TEMPLATE_REGISTRY["api_maintainer"]
    names = {c.name for c in t.capabilities}
    assert "code_read" in names
    assert "code_write" in names
    assert "pr_create" in names


def test_coordinator_template_no_code_write() -> None:
    t = TEMPLATE_REGISTRY["coordinator"]
    names = {c.name for c in t.capabilities}
    assert "code_write" not in names
    assert "graph_query_global" in names


def test_infra_template_auto_merge_disabled() -> None:
    t = TEMPLATE_REGISTRY["infra_maintainer"]
    assert t.governance.auto_merge_allowed is False


# ── AgentScope ─────────────────────────────────────────────────────────


def test_agent_scope_defaults() -> None:
    scope = AgentScope()
    assert scope.allowed_paths == []
    assert scope.allowed_extensions == []
    assert scope.deny_paths == []


def test_agent_scope_with_values() -> None:
    scope = AgentScope(
        allowed_paths=["src/payments/", "src/billing/"],
        allowed_extensions=[".py"],
        deny_paths=["src/payments/legacy/"],
    )
    assert len(scope.allowed_paths) == 2
    assert scope.deny_paths == ["src/payments/legacy/"]


# ── RepositoryRegistration ─────────────────────────────────────────────


def _make_reg(**kwargs: object) -> RepositoryRegistration:
    defaults: dict[str, object] = dict(
        name="my-api",
        git_url="https://github.com/org/my-api",
        git_provider=GitProviderConfig(
            provider_type="github",
            credentials_secret_ref="env:GITHUB_TOKEN",
        ),
        primary_language="python",
    )
    defaults.update(kwargs)
    return RepositoryRegistration.model_validate(defaults)


def test_registration_defaults() -> None:
    reg = _make_reg()
    assert reg.agent_template == "api_maintainer"
    assert reg.autonomy_level is AutonomyLevel.SUPERVISED
    assert reg.scope is None
    assert reg.dependencies == []


def test_registration_with_scope() -> None:
    reg = _make_reg(
        scope=AgentScope(
            allowed_paths=["src/"],
            allowed_extensions=[".py"],
            deny_paths=[],
        )
    )
    assert reg.scope is not None
    assert reg.scope.allowed_paths == ["src/"]


def test_registration_ignore_patterns_default() -> None:
    reg = _make_reg()
    assert "node_modules/**" in reg.ignore_patterns
    assert ".git/**" in reg.ignore_patterns


def test_registration_ignore_patterns_independent() -> None:
    """Each instance must have its own copy of ignore_patterns."""
    r1 = _make_reg()
    r2 = _make_reg()
    r1.ignore_patterns.append("custom/**")
    assert "custom/**" not in r2.ignore_patterns


# ── AgentInstanceConfig.from_registration ─────────────────────────────


def test_from_registration_basic() -> None:
    reg = _make_reg()
    config = AgentInstanceConfig.from_registration(reg, agent_id="agent-1")
    assert config.agent_id == "agent-1"
    assert config.assigned_repo == "my-api"
    assert config.template_id == "api_maintainer"
    assert config.status is AgentStatus.DORMANT


def test_from_registration_capability_composition() -> None:
    extra_cap = Capability(name="custom_tool", description="A custom tool")
    reg = _make_reg(additional_capabilities=[extra_cap])
    config = AgentInstanceConfig.from_registration(reg, agent_id="a1")
    names = {c.name for c in config.capabilities}
    assert "custom_tool" in names
    assert "code_read" in names


def test_from_registration_capability_override_removes() -> None:
    reg = _make_reg(capability_overrides=["code_write"])
    config = AgentInstanceConfig.from_registration(reg, agent_id="a1")
    names = {c.name for c in config.capabilities}
    assert "code_write" not in names
    assert "code_read" in names


def test_from_registration_governance_override() -> None:
    reg = _make_reg(governance_overrides={"max_files_per_change": 5})
    config = AgentInstanceConfig.from_registration(reg, agent_id="a1")
    assert config.governance.max_files_per_change == 5


def test_from_registration_context_override() -> None:
    reg = _make_reg(context_overrides={"max_context_tokens": 64_000})
    config = AgentInstanceConfig.from_registration(reg, agent_id="a1")
    assert config.context_config.max_context_tokens == 64_000


def test_from_registration_unknown_template_raises() -> None:
    reg = _make_reg(agent_template="nonexistent_template")
    with pytest.raises(KeyError):
        AgentInstanceConfig.from_registration(reg, agent_id="a1")


# ── AgentReactivityConfig ──────────────────────────────────────────────


def test_reactivity_config_defaults() -> None:
    r = AgentReactivityConfig()
    assert r.wake_on_direct_change is True
    assert r.wake_on_transitive_change is False
    assert r.catchup_strategy == "full_diff"
    assert r.debounce_seconds == 30
    assert r.rebase_strategy == "rebase"


def test_reactivity_config_invalid_strategy() -> None:
    with pytest.raises(ValidationError):
        AgentReactivityConfig(catchup_strategy="guess")  # type: ignore[arg-type]
