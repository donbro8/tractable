"""Git provider factory.

Returns a concrete GitProvider implementation for a given GitProviderConfig.

Source: tech-spec.py §3.3 — GitProviderConfig
"""

from __future__ import annotations

from tractable.errors import RecoverableError
from tractable.protocols.git_provider import GitProvider
from tractable.types.config import GitProviderConfig


def create_git_provider(config: GitProviderConfig) -> GitProvider:
    """Instantiate and return the appropriate GitProvider for *config*.

    Raises:
        RecoverableError: if ``config.provider_type`` is not yet supported.
    """
    if config.provider_type == "github":
        from tractable.providers.github import GitHubProvider

        return GitHubProvider(config)
    raise RecoverableError(
        f"Git provider '{config.provider_type}' is not yet implemented. "
        "Currently only 'github' is supported."
    )
