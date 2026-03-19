"""Example registration config for a Python API repository.

Run with:
    tractable register examples/register_python_api.py

Requires:
    - GITHUB_TOKEN environment variable with repo scope
    - FalkorDB running (docker compose -f deploy/docker-compose.yml up)
    - DATABASE_URL pointing to a running PostgreSQL instance
"""

from tractable.types.config import (
    GitProviderConfig,
    RepositoryRegistration,
)

registration = RepositoryRegistration(
    name="psf/requests",
    git_url="https://github.com/psf/requests.git",
    git_provider=GitProviderConfig(
        provider_type="github",
        credentials_secret_ref="GITHUB_TOKEN",
        default_branch="main",
    ),
    primary_language="python",
    agent_template="api_maintainer",
)
