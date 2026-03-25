"""Example registration config for a TypeScript/React frontend repository.

Run with:
    tractable register examples/register_typescript_frontend.py

Requires:
    - GITHUB_TOKEN environment variable with repo scope
    - FalkorDB running (docker compose -f deploy/docker-compose.yml up)
    - DATABASE_URL pointing to a running PostgreSQL instance
"""

from tractable.types.config import (
    AgentScope,
    GitProviderConfig,
    GovernancePolicy,
    RepositoryRegistration,
    SensitivePathRule,
)

registration = RepositoryRegistration(
    name="my-org/my-frontend",
    git_url="https://github.com/my-org/my-frontend.git",
    git_provider=GitProviderConfig(
        provider_type="github",
        credentials_secret_ref="GITHUB_TOKEN",
        default_branch="main",
    ),
    primary_language="typescript",
    agent_template="frontend_maintainer",
    scope=AgentScope(
        allowed_paths=["src/"],
    ),
    governance_overrides=GovernancePolicy(
        requires_tests_pass=True,
        requires_lint_pass=True,
        sensitive_path_patterns=[
            SensitivePathRule(
                pattern="**/public/**",
                reason="Static assets managed separately",
                policy="human_review_always",
            ),
        ],
    ).model_dump(),
)
