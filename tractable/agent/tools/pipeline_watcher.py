"""pipeline_watcher tool — monitor CI check runs on a pull request.

TASK-2.4.5: Implements the ``pipeline_watcher`` tool satisfying the ``Tool``
Protocol (tech-spec.py §2.6).  Operation: ``get_check_status``.

Fetches all CI check runs for the PR's head commit via ``GitProvider``.
If any check has ``conclusion="failure"``, also fetches the first 4000
characters of its log and includes them in ``output["failure_logs"]``.

Sources:
- tech-spec.py §2.6 — Tool Protocol
- tech-spec.py §2.1 — GitProvider Protocol
- CLAUDE.md        — Error taxonomy, structured logging
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from tractable.protocols.tool import ToolResult

if TYPE_CHECKING:
    from tractable.protocols.git_provider import GitProvider

_log = structlog.get_logger()

_LOG_FETCH_LIMIT = 4000


class PipelineWatcherTool:
    """Query CI check run status for an open pull request.

    Parameters
    ----------
    git_provider:
        Provider implementation satisfying the GitProvider Protocol.
    repo_id:
        Repository identifier passed to the GitProvider (e.g. ``"org/repo"``).
    agent_id:
        Identifier of the running agent (included in every log entry).
    task_id:
        Identifier of the current task (included in every log entry).
    repo:
        Human-readable repository name (included in every log entry).
    """

    def __init__(
        self,
        git_provider: GitProvider,
        repo_id: str,
        agent_id: str,
        task_id: str,
        repo: str,
    ) -> None:
        self._git_provider = git_provider
        self._repo_id = repo_id
        self._agent_id = agent_id
        self._task_id = task_id
        self._repo = repo

    # ── Tool Protocol ────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "pipeline_watcher"

    @property
    def description(self) -> str:
        return (
            "Check the CI status of a pull request. "
            "Returns check_runs list, all_passed bool, any_failed bool. "
            "When any check failed, also returns failure_logs (first 4000 chars)."
        )

    async def invoke(self, params: dict[str, Any]) -> ToolResult:
        """Dispatch to the appropriate pipeline operation.

        Expected ``params`` keys:

        - ``operation``: ``"get_check_status"``
        - ``pr_number``: PR number (int) to check
        """
        operation: str = params.get("operation", "")

        if operation == "get_check_status":
            return await self._get_check_status(params)

        return ToolResult(success=False, error=f"Unknown operation: {operation!r}")

    # ── Operations ───────────────────────────────────────────────────────────

    async def _get_check_status(self, params: dict[str, Any]) -> ToolResult:
        pr_number_raw: Any = params.get("pr_number", 0)
        pr_number: int = int(pr_number_raw)

        check_runs = await self._git_provider.get_check_runs(self._repo_id, pr_number)

        run_list: list[dict[str, Any]] = [
            {
                "name": r.name,
                "status": r.status,
                "conclusion": r.conclusion,
                "log_url": r.log_url,
            }
            for r in check_runs
        ]

        all_passed: bool = bool(check_runs) and all(
            r.conclusion == "success" for r in check_runs
        )
        any_failed: bool = any(r.conclusion == "failure" for r in check_runs)

        output: dict[str, Any] = {
            "check_runs": run_list,
            "all_passed": all_passed,
            "any_failed": any_failed,
        }

        if any_failed:
            # Fetch log for the first failed check run.
            failed_run = next(r for r in check_runs if r.conclusion == "failure")
            log_text: str = ""
            if failed_run.log_url:
                raw_log = await self._git_provider.get_check_run_log(failed_run.log_url)
                log_text = raw_log[:_LOG_FETCH_LIMIT]
            output["failure_logs"] = log_text

        _log.info(
            "check_status_fetched",
            level="info",
            agent_id=self._agent_id,
            task_id=self._task_id,
            repo=self._repo,
            pr_number=pr_number,
            all_passed=all_passed,
            any_failed=any_failed,
        )

        return ToolResult(success=True, output=output)
