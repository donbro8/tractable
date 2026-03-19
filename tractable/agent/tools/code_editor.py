"""code_editor tool — file read/write/list with scope and sensitive-path enforcement.

TASK-2.4.1: Implements the ``code_editor`` tool satisfying the ``Tool`` Protocol
(tech-spec.py §2.6).  All paths are resolved relative to ``working_dir`` (the
cloned repo root).  Scope enforcement and sensitive-path checks gate every
operation; violations raise ``GovernanceError`` and append an ``AuditEntry``.

Sources:
- tech-spec.py §2.6 — Tool Protocol
- tech-spec.py §3  — GovernancePolicy, SensitivePathRule, AgentScope
- CLAUDE.md        — Governance enforcement points
"""

from __future__ import annotations

import fnmatch
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from tractable.errors import GovernanceError
from tractable.protocols.agent_state_store import AgentStateStore
from tractable.protocols.tool import ToolResult
from tractable.types.agent import AuditEntry
from tractable.types.config import AgentScope, GovernancePolicy

_log = structlog.get_logger()


def _path_matches_prefix(resolved_str: str, prefix: str) -> bool:
    """Return True if ``resolved_str`` equals ``prefix`` or is directly inside it.

    A plain ``startswith`` check is insufficient because it would match
    ``/repo/src_extra/file.py`` against the prefix ``/repo/src``.  We require
    that the next character after ``prefix`` is the OS path separator (meaning
    ``resolved_str`` is a child) or that the strings are equal (meaning
    ``resolved_str`` IS the prefixed path itself).
    """
    if resolved_str == prefix:
        return True
    return resolved_str.startswith(prefix + os.sep)


class CodeEditorTool:
    """File editor tool with scope and sensitive-path enforcement.

    Parameters
    ----------
    working_dir:
        Absolute path to the cloned repository root.  All relative paths
        provided by the agent are resolved against this directory.
    scope:
        Optional path-based restrictions.  ``deny_paths`` takes precedence
        over ``allowed_paths``; both are prefix-matched against the resolved
        absolute path string.
    governance:
        GovernancePolicy whose ``sensitive_path_patterns`` are checked on
        every ``write_file`` call after scope enforcement passes.
    state_store:
        Used to append ``AuditEntry`` records on governance violations.
    agent_id:
        Identifier of the running agent (included in every log entry and
        audit record).
    task_id:
        Identifier of the current task (included in every log entry and
        audit record).
    repo:
        Human-readable repository name (included in every log entry).
    """

    def __init__(
        self,
        working_dir: Path,
        scope: AgentScope,
        governance: GovernancePolicy,
        state_store: AgentStateStore,
        agent_id: str,
        task_id: str,
        repo: str,
    ) -> None:
        self._working_dir = working_dir.resolve()
        self._scope = scope
        self._governance = governance
        self._state_store = state_store
        self._agent_id = agent_id
        self._task_id = task_id
        self._repo = repo

    # ── Tool Protocol ───────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "code_editor"

    @property
    def description(self) -> str:
        return (
            "Read, write, or list files within the assigned repository. "
            "All operations are scoped to the cloned repository root and "
            "subject to governance policy enforcement."
        )

    async def invoke(self, params: dict[str, Any]) -> ToolResult:
        """Dispatch to the appropriate file operation.

        Expected ``params`` keys:
        - ``operation``: one of ``"read_file"``, ``"write_file"``, ``"list_files"``
        - ``path``: target file or directory path (relative to working dir)
        - ``content``: file content string (``write_file`` only)
        """
        operation: str = params.get("operation", "")
        path: str = params.get("path", "")

        if operation == "read_file":
            return await self._read_file(path)
        if operation == "write_file":
            content: str = params.get("content", "")
            return await self._write_file(path, content)
        if operation == "list_files":
            return await self._list_files(path)

        return ToolResult(success=False, error=f"Unknown operation: {operation!r}")

    # ── Operations ──────────────────────────────────────────────────────────

    async def _read_file(self, path: str) -> ToolResult:
        resolved = self._resolve(path)
        self._check_traversal(path, resolved)
        self._check_deny(resolved, operation="read_file")
        return ToolResult(success=True, output=resolved.read_text(encoding="utf-8"))

    async def _write_file(self, path: str, content: str) -> ToolResult:
        resolved = self._resolve(path)
        self._check_traversal(path, resolved)
        self._check_deny(resolved, operation="write_file")
        self._check_allowed(resolved, operation="write_file")
        self._check_sensitive(resolved)

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        bytes_written = len(content.encode("utf-8"))

        _log.info(
            "file_written",
            level="info",
            agent_id=self._agent_id,
            task_id=self._task_id,
            repo=self._repo,
            file_path=str(resolved),
            bytes_written=bytes_written,
        )

        return ToolResult(success=True)

    async def _list_files(self, directory: str) -> ToolResult:
        resolved = self._resolve(directory)
        self._check_traversal(directory, resolved)
        self._check_deny(resolved, operation="list_files")
        self._check_allowed(resolved, operation="list_files")

        if not resolved.is_dir():
            return ToolResult(success=False, error=f"Not a directory: {directory!r}")

        paths = [str(p.relative_to(self._working_dir)) for p in resolved.iterdir()]
        return ToolResult(success=True, output=paths)

    # ── Enforcement helpers ─────────────────────────────────────────────────

    def _resolve(self, path: str) -> Path:
        """Resolve a relative path against the working directory."""
        return (self._working_dir / path).resolve()

    def _check_traversal(self, path: str, resolved: Path) -> None:
        """Raise GovernanceError if the path contains '..' or escapes working_dir.

        We reject '..' components explicitly (security-in-depth) so that
        well-formed allowlist patterns cannot be bypassed by a crafted path
        that happens to resolve back inside the working directory.
        """
        # Reject any path containing '..' components regardless of where it resolves.
        pure = Path(path)
        if any(part == ".." for part in pure.parts):
            raise GovernanceError(
                f"Path traversal detected: {path!r} contains '..' components"
            )
        # Belt-and-suspenders: also verify the resolved path is inside working_dir.
        try:
            resolved.relative_to(self._working_dir)
        except ValueError as exc:
            raise GovernanceError(
                f"Path traversal detected: {resolved!r} escapes working directory "
                f"{self._working_dir!r}"
            ) from exc

    def _check_deny(self, resolved: Path, *, operation: str) -> None:
        """Raise GovernanceError if ``resolved`` matches any deny_paths entry."""
        resolved_str = str(resolved)
        for deny_pattern in self._scope.deny_paths:
            deny_resolved = str((self._working_dir / deny_pattern).resolve())
            if _path_matches_prefix(resolved_str, deny_resolved) or fnmatch.fnmatch(
                resolved_str, deny_resolved
            ):
                self._emit_blocked_log(resolved_str, reason="deny_paths")
                self._append_audit_sync(
                    action="scope_violation",
                    file_path=resolved_str,
                )
                raise GovernanceError(
                    f"Path {resolved_str!r} matches deny_paths rule "
                    f"{deny_pattern!r} (operation={operation!r})"
                )

    def _check_allowed(self, resolved: Path, *, operation: str) -> None:
        """Raise GovernanceError if ``resolved`` is outside allowed_paths.

        If ``allowed_paths`` is empty, all paths within ``working_dir`` are
        allowed (the deny check is the only gate).
        """
        if not self._scope.allowed_paths:
            return

        resolved_str = str(resolved)
        for allow_pattern in self._scope.allowed_paths:
            allow_resolved = str((self._working_dir / allow_pattern).resolve())
            if _path_matches_prefix(resolved_str, allow_resolved):
                return

        self._emit_blocked_log(resolved_str, reason="outside_allowed_paths")
        self._append_audit_sync(
            action="scope_violation",
            file_path=resolved_str,
        )
        raise GovernanceError(
            f"Path {resolved_str!r} is not within allowed_paths "
            f"{self._scope.allowed_paths!r} (operation={operation!r})"
        )

    def _check_sensitive(self, resolved: Path) -> None:
        """Raise GovernanceError if ``resolved`` matches a sensitive path rule."""
        resolved_str = str(resolved)
        for rule in self._governance.sensitive_path_patterns:
            # Match against both the full resolved path and the path relative
            # to working_dir so that glob patterns like "src/auth/**" work.
            rel_path = str(resolved.relative_to(self._working_dir))
            if fnmatch.fnmatch(resolved_str, str(self._working_dir / rule.pattern)) or \
               fnmatch.fnmatch(rel_path, rule.pattern):
                self._emit_blocked_log(resolved_str, reason="sensitive_path")
                self._append_audit_sync(
                    action="sensitive_path_blocked",
                    file_path=resolved_str,
                )
                raise GovernanceError(
                    f"Path {resolved_str!r} matches sensitive path rule "
                    f"{rule.pattern!r} (reason={rule.reason!r})",
                )

    def _emit_blocked_log(self, file_path: str, *, reason: str) -> None:
        _log.warning(
            "write_blocked",
            level="warning",
            agent_id=self._agent_id,
            task_id=self._task_id,
            repo=self._repo,
            file_path=file_path,
            reason=reason,
        )

    def _append_audit_sync(self, *, action: str, file_path: str) -> None:
        """Schedule audit entry append.

        Because ``_check_*`` helpers are called from sync context inside the
        async ``invoke`` method (before any ``await``), we store a pending
        entry that ``invoke`` flushes.  In practice the helpers always run
        inside an already-running event loop, so we use a small helper that
        the caller (``invoke``) awaits via a deferred list.  However, since
        the GovernanceError is raised immediately after this call, we instead
        use ``asyncio.ensure_future`` to fire-and-forget the coroutine.
        This is acceptable because audit entries must not be lost even when
        execution is halted by the exception.

        Implementation note: we cannot ``await`` inside the sync helper, so
        we use ``asyncio.get_event_loop().create_task`` which is always
        available inside an async execution context.
        """
        import asyncio

        entry = AuditEntry(
            timestamp=datetime.now(UTC),
            agent_id=self._agent_id,
            task_id=self._task_id,
            action=action,
            detail={"file_path": file_path},
            outcome="failure",
        )
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._state_store.append_audit_entry(entry))
        except RuntimeError:
            # No running event loop — emit a critical log so the audit trail is
            # never silently lost even if the store write cannot be scheduled.
            _log.critical(
                "audit_entry_lost",
                agent_id=self._agent_id,
                task_id=self._task_id,
                action=action,
                file_path=file_path,
                reason="no_event_loop",
            )
