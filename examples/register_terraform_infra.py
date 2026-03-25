"""Example registration config for a Terraform infrastructure repository.

Run with:
    tractable register examples/register_terraform_infra.py

Requires:
    - GITHUB_TOKEN environment variable with repo scope
    - FalkorDB running (docker compose -f deploy/docker-compose.yml up)
    - DATABASE_URL pointing to a running PostgreSQL instance
"""

from tractable.types.config import (
    INFRA_MAINTAINER_TEMPLATE,
    GitProviderConfig,
    GovernancePolicy,
    RepositoryRegistration,
)
from tractable.types.enums import AutonomyLevel

registration = RepositoryRegistration(
    name="my-org/my-infra",
    git_url="https://github.com/my-org/my-infra.git",
    git_provider=GitProviderConfig(
        provider_type="github",
        credentials_secret_ref="GITHUB_TOKEN",
        default_branch="main",
    ),
    primary_language="hcl",
    agent_template="infra_maintainer",
    autonomy_level=AutonomyLevel.SUPERVISED,
    governance_overrides=GovernancePolicy(
        auto_merge_allowed=False,
        sensitive_path_patterns=INFRA_MAINTAINER_TEMPLATE.governance.sensitive_path_patterns,
    ).model_dump(),
)
