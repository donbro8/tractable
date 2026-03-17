"""Error taxonomy for the Tractable framework.

All errors are one of four types:
- TransientError  — retry with exponential backoff (max 3)
- RecoverableError — re-plan the current task phase
- GovernanceError  — halt, write audit entry, notify human
- FatalError       — fail task gracefully, preserve checkpoint

Source: CLAUDE.md — Error taxonomy
"""

from __future__ import annotations


class TractableError(Exception):
    """Base class for all Tractable errors."""


class TransientError(TractableError):
    """Transient failure — retry with exponential backoff (max 3 retries).

    Raised for recoverable infrastructure failures such as API rate limits
    or temporary network errors.
    """

    def __init__(self, message: str, retry_after: int = 60) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RecoverableError(TractableError):
    """Recoverable failure — re-plan the current task phase.

    Raised when the operation cannot succeed in its current form but a
    different plan or approach may succeed (e.g., resource not found,
    merge conflict).
    """


class GovernanceError(TractableError):
    """Governance violation — halt execution, write audit entry, notify human.

    Raised when an agent attempts an action that violates configured
    governance policies (scope, sensitive paths, approval requirements).
    Written to both structlog output and the append-only AuditEntry store.
    """


class FatalError(TractableError):
    """Fatal failure — fail task gracefully, preserve checkpoint.

    Raised for unrecoverable errors where the task cannot continue under
    any circumstances. The agent checkpoints its state before exiting.
    """
