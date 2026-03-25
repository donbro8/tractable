"""GitHubProvider — concrete implementation of GitProvider for GitHub.

Implements read operations only (Phase 1). Write operations
(create_branch, create_pull_request, merge_pull_request, set_branch_protection)
are defined as stubs that raise RecoverableError.

Uses httpx for GitHub REST API calls and git subprocess for cloning.

All errors follow the four-class taxonomy from ``tractable.errors``:
- 401/403 authentication → FatalError
- 429 / 5xx → TransientError
- 404 → RecoverableError
- subprocess clone failure → TransientError

Source: tech-spec.py §2.1 — GitProvider Protocol
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import urllib.parse
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any, Literal, cast

import httpx
import structlog

from tractable.errors import FatalError, GovernanceError, RecoverableError, TransientError
from tractable.types.config import GitProviderConfig
from tractable.types.git import (
    BranchProtectionRules,
    CheckRunInfo,
    CommitEntry,
    FileEntry,
    MergeResult,
    PullRequestHandle,
)

logger = structlog.get_logger(__name__)

_GITHUB_API_BASE = "https://api.github.com"

# Characters that must not appear in a repository path passed to subprocess.
# Covers common shell metacharacters per phase-1-analysis.md §3.6 and §4.3.
_SHELL_META: frozenset[str] = frozenset(
    [";", "&", "|", "$", "(", ")", "<", ">", "\\", '"', "'", "`", "\x00"]
)


def _validate_repo_url(repo_id: str) -> str:
    """Validate *repo_id* and return a normalised ``"org/repo"`` path.

    Accepts two input formats:

    - Full HTTPS URL: ``"https://github.com/org/repo.git"``
    - Shorthand path: ``"org/repo"``

    Validation rules (from phase-1-analysis.md §4.3):

    - For full URLs: scheme must be ``https``; host must be in the
      ``TRACTABLE_ALLOWED_GIT_HOSTS`` allowlist (default:
      ``github.com,api.github.com``); path must not contain ``..``, null
      bytes, or shell metacharacters.
    - For shorthand paths: same metacharacter restrictions apply.

    Raises:
        GovernanceError: if any validation rule is violated.
    """
    allowed_raw = os.environ.get(
        "TRACTABLE_ALLOWED_GIT_HOSTS", "github.com,api.github.com"
    )
    allowed_hosts: frozenset[str] = frozenset(
        h.strip() for h in allowed_raw.split(",") if h.strip()
    )

    if "://" in repo_id:
        parsed = urllib.parse.urlparse(repo_id)
        if parsed.scheme != "https":
            raise GovernanceError(
                f"URL validation failed: scheme must be 'https', got '{parsed.scheme}'"
            )
        hostname: str = parsed.hostname or ""
        if hostname not in allowed_hosts:
            raise GovernanceError(
                f"URL validation failed: host '{hostname}' is not in the allowed list"
            )
        path: str = parsed.path
        if ".." in path or any(c in path for c in _SHELL_META):
            raise GovernanceError(
                "URL validation failed: repository path contains invalid characters"
            )
        # Normalise to "org/repo": strip leading "/" and trailing ".git"
        clean = path.lstrip("/")
        if clean.endswith(".git"):
            clean = clean[:-4]
        return clean
    else:
        # Shorthand "org/repo" format — validate path component only.
        if ".." in repo_id or any(c in repo_id for c in _SHELL_META):
            raise GovernanceError(
                "URL validation failed: repository ID contains invalid characters"
            )
        return repo_id


class GitHubProvider:
    """Concrete implementation of the GitProvider Protocol for GitHub.

    Only read operations are implemented (Phase 1). Write operations raise
    ``NotImplementedError("Implemented in Phase 2")``.

    The GitHub token is resolved from the environment variable named by
    ``config.credentials_secret_ref`` at the moment the credential is needed
    (inside ``clone()``, ``create_pull_request()``, etc.), not at construction
    time.  The resolved token string is never stored in an instance attribute
    and is never passed to any logging call.
    """

    def __init__(self, config: GitProviderConfig | None = None) -> None:
        if config is None:
            config = GitProviderConfig(
                provider_type="github",
                credentials_secret_ref="GITHUB_TOKEN",
            )
        self._credentials_secret_ref = config.credentials_secret_ref
        self._base_url = (config.base_url or _GITHUB_API_BASE).rstrip("/")
        self._default_branch = config.default_branch

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_token(self) -> str:
        """Resolve the GitHub token from the environment at call time.

        The token value is never stored as an instance attribute.
        """
        token = os.environ.get(self._credentials_secret_ref)
        if token is None:
            logger.error(
                "github.auth_failure",
                credential_var=self._credentials_secret_ref,
            )
            raise FatalError(
                f"GitHub credentials not found: environment variable "
                f"'{self._credentials_secret_ref}' is not set. "
                "Set this variable to a GitHub personal access token with 'repo' scope."
            )
        return token

    def _make_headers(self) -> dict[str, str]:
        """Build authorization headers, resolving the token at call time."""
        token = self._get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _make_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self._base_url, headers=self._make_headers())

    def _handle_response_errors(self, response: httpx.Response, context: str) -> None:
        """Convert GitHub HTTP error codes to domain errors.

        Mapping:
        - 401      → FatalError  (invalid credentials, cannot recover)
        - 403, 429 → TransientError  (rate limit, retry after delay)
        - 404      → RecoverableError  (resource not found)
        - 500, 503 → TransientError  (server error, retryable)
        """
        status = response.status_code

        if status == 401:
            logger.error(
                "github.auth_invalid",
                context=context,
                status=status,
            )
            raise FatalError(
                f"GitHub API authentication failed ({context}): "
                "token is invalid or lacks required scopes"
            )

        if status in (403, 429):
            retry_after = int(response.headers.get("Retry-After", "60"))
            logger.warning(
                "github.rate_limit",
                context=context,
                status=status,
                retry_after=retry_after,
            )
            raise TransientError(
                f"GitHub API rate limit exceeded ({context})",
                retry_after=retry_after,
            )

        if status == 404:
            raise RecoverableError(f"GitHub resource not found: {context}")

        if status in (500, 503):
            logger.warning(
                "github.server_error",
                context=context,
                status=status,
            )
            raise TransientError(
                f"GitHub API server error {status} ({context})"
            )

        response.raise_for_status()

    # ── Read operations ───────────────────────────────────────────────────

    async def clone(
        self,
        repo_id: str,
        target_path: str,
        branch: str = "main",
        sparse_paths: Sequence[str] | None = None,
        post_clone_fn: Callable[[str], None] | None = None,
    ) -> str:
        """Clone or sparse-checkout a repository. Returns local path.

        *repo_id* may be either a full HTTPS URL
        (``"https://github.com/org/repo.git"``) or a shorthand path
        (``"org/repo"``).  The input is validated against scheme, host
        allowlist, and shell-metacharacter rules before the credentialed
        clone URL is constructed.  The token is NEVER written to log output.

        If *post_clone_fn* is provided it is called with *target_path* after a
        successful clone.  If *post_clone_fn* raises (or if the clone itself
        fails), *target_path* is removed via ``shutil.rmtree`` before the
        exception propagates, preventing orphaned temp directories.
        """
        repo_path = _validate_repo_url(repo_id)
        log = logger.bind(repo=repo_path, branch=branch)
        log.info("git.clone.start")

        # Resolve token at call time — never store it or log it.
        token = self._get_token()

        # Plain HTTPS URL — credentials are injected via the subprocess
        # environment using git's extraheader mechanism, not via the URL.
        clone_url = f"https://github.com/{repo_path}.git"

        # Build subprocess environment that injects the Authorization header
        # for github.com via GIT_CONFIG_COUNT.  The token value is only in
        # the environment mapping and never appears in the argv list passed
        # to subprocess.run().
        _credentials_b64 = base64.b64encode(
            f"x-access-token:{token}".encode()
        ).decode()
        _clone_env: dict[str, str] = {
            **os.environ,
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"Authorization: Basic {_credentials_b64}",
        }

        try:
            if sparse_paths:
                result: subprocess.CompletedProcess[str] = subprocess.run(
                    [
                        "git", "clone",
                        "--filter=blob:none", "--sparse",
                        "--branch", branch,
                        clone_url, target_path,
                    ],
                    capture_output=True,
                    text=True,
                    env=_clone_env,
                )
                if result.returncode != 0:
                    raise TransientError(
                        f"Failed to clone {repo_path} (branch={branch}): "
                        f"git exited {result.returncode}"
                    )
                sparse_result: subprocess.CompletedProcess[str] = subprocess.run(
                    ["git", "sparse-checkout", "set", *sparse_paths],
                    capture_output=True,
                    text=True,
                    cwd=target_path,
                )
                if sparse_result.returncode != 0:
                    raise TransientError(
                        f"Failed to configure sparse checkout for {repo_path}: "
                        f"git exited {sparse_result.returncode}"
                    )
            else:
                result = subprocess.run(
                    ["git", "clone", "--branch", branch, clone_url, target_path],
                    capture_output=True,
                    text=True,
                    env=_clone_env,
                )
                if result.returncode != 0:
                    raise TransientError(
                        f"Failed to clone {repo_path} (branch={branch}): "
                        f"git exited {result.returncode}"
                    )

            if post_clone_fn is not None:
                post_clone_fn(target_path)

        except Exception:
            shutil.rmtree(target_path, ignore_errors=True)
            raise

        log.info("git.clone.done")
        return target_path

    async def get_file_content(
        self,
        repo_id: str,
        file_path: str,
        ref: str = "main",
    ) -> bytes:
        """Read a single file at a given ref. Handles files up to 1 MB."""
        async with self._make_client() as client:
            response = await client.get(
                f"/repos/{repo_id}/contents/{file_path}",
                params={"ref": ref},
            )
        self._handle_response_errors(response, f"{repo_id}/{file_path}@{ref}")
        data: Any = response.json()
        raw_content: Any = data.get("content", "")
        if not isinstance(raw_content, str):
            raw_content = ""
        return base64.b64decode(raw_content)

    async def list_files(
        self,
        repo_id: str,
        path: str = "",
        ref: str = "main",
    ) -> Sequence[FileEntry]:
        """List files and directories at *path* for the given ref."""
        async with self._make_client() as client:
            response = await client.get(
                f"/repos/{repo_id}/contents/{path}",
                params={"ref": ref},
            )
        self._handle_response_errors(response, f"{repo_id}/{path or '<root>'}@{ref}")
        items: Any = response.json()
        entries: list[FileEntry] = []
        for item in items:
            item_path: Any = item.get("path", "")
            item_type: Any = item.get("type", "")
            item_size: Any = item.get("size")
            item_sha: Any = item.get("sha")
            entries.append(
                FileEntry(
                    path=str(item_path),
                    is_directory=(item_type == "dir"),
                    size_bytes=int(item_size) if isinstance(item_size, int) else None,
                    sha=str(item_sha) if item_sha is not None else None,
                )
            )
        return entries

    async def get_diff(
        self,
        repo_id: str,
        base_ref: str,
        head_ref: str,
    ) -> str:
        """Return the unified diff between two refs."""
        async with self._make_client() as client:
            response = await client.get(
                f"/repos/{repo_id}/compare/{base_ref}...{head_ref}",
                headers={"Accept": "application/vnd.github.diff"},
            )
        self._handle_response_errors(response, f"{repo_id} {base_ref}...{head_ref}")
        return response.text

    async def get_commit_history(
        self,
        repo_id: str,
        path: str | None = None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> Sequence[CommitEntry]:
        """Return commit history, optionally filtered by path and/or since timestamp."""
        params: dict[str, str] = {"per_page": str(limit)}
        if path is not None:
            params["path"] = path
        if since is not None:
            params["since"] = since.isoformat()

        async with self._make_client() as client:
            response = await client.get(
                f"/repos/{repo_id}/commits",
                params=params,
            )
        self._handle_response_errors(response, f"{repo_id} commit history")

        raw_commits: Any = response.json()
        commits: list[CommitEntry] = []
        for c in raw_commits:
            c_dict = cast(dict[str, Any], c)
            commit_meta_raw: Any = c_dict.get("commit", {})
            if not isinstance(commit_meta_raw, dict):
                continue
            commit_meta = cast(dict[str, Any], commit_meta_raw)
            author_raw: Any = commit_meta.get("author") or {}
            author_meta = cast(dict[str, Any], author_raw) if isinstance(author_raw, dict) else {}
            raw_date: str = str(author_meta.get("date") or "1970-01-01T00:00:00Z")
            raw_files: Any = c_dict.get("files", [])
            files_changed: list[str] = []
            raw_files_list: list[Any] = (
                cast(list[Any], raw_files) if isinstance(raw_files, list) else []
            )
            for f_item in raw_files_list:
                f_dict = cast(dict[str, Any], f_item) if isinstance(f_item, dict) else {}
                fname: Any = f_dict.get("filename", "")
                if isinstance(fname, str) and fname:
                    files_changed.append(fname)
            commits.append(
                CommitEntry(
                    sha=str(c_dict.get("sha") or ""),
                    message=str(commit_meta.get("message") or ""),
                    author=str(author_meta.get("name") or ""),
                    timestamp=datetime.fromisoformat(raw_date.replace("Z", "+00:00")),
                    files_changed=files_changed,
                )
            )
        return commits

    # ── Write operations (Phase 2 stubs) ─────────────────────────────────

    async def create_branch(
        self,
        repo_id: str,
        branch_name: str,
        from_ref: str = "main",
    ) -> str:
        async with self._make_client() as client:
            check_resp = await client.get(f"/repos/{repo_id}/git/refs/heads/{branch_name}")
            if check_resp.status_code == 200:
                raise RecoverableError(f"Branch already exists: {branch_name}")
            elif check_resp.status_code != 404:
                self._handle_response_errors(check_resp, f"check branch {branch_name} in {repo_id}")

            ref_resp = await client.get(f"/repos/{repo_id}/git/refs/heads/{from_ref}")
            self._handle_response_errors(ref_resp, f"get {from_ref} for {repo_id}")
            from_sha: str = ref_resp.json().get("object", {}).get("sha", "")
            if not from_sha:
                raise RecoverableError(f"Could not resolve SHA for {from_ref} in {repo_id}")

            create_resp = await client.post(
                f"/repos/{repo_id}/git/refs",
                json={"ref": f"refs/heads/{branch_name}", "sha": from_sha},
            )
            self._handle_response_errors(create_resp, f"create branch {branch_name} in {repo_id}")
            sha = str(create_resp.json().get("object", {}).get("sha", ""))
            logger.info(
                "branch_created",
                repo=repo_id,
                branch_name=branch_name,
                sha=sha,
            )
            return sha

    async def create_pull_request(
        self,
        repo_id: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
        reviewers: Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
    ) -> PullRequestHandle:
        payload = {
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base_branch,
        }
        async with self._make_client() as client:
            resp = await client.post(f"/repos/{repo_id}/pulls", json=payload)
            self._handle_response_errors(resp, f"create PR in {repo_id}")

            data: dict[str, Any] = resp.json()
            pr_number: int = data.get("number", 0)
            url: str = data.get("html_url", "")

            handle = PullRequestHandle(
                provider="github",
                repo_id=repo_id,
                pr_number=pr_number,
                url=url,
                head_branch=head_branch,
                base_branch=base_branch,
            )

            logger.info(
                "pull_request_created",
                repo=repo_id,
                pr_number=pr_number,
                head_branch=head_branch,
                base_branch=base_branch,
            )

            if reviewers:
                rev_resp = await client.post(
                    f"/repos/{repo_id}/pulls/{pr_number}/requested_reviewers",
                    json={"reviewers": list(reviewers)},
                )
                self._handle_response_errors(
                    rev_resp, f"request reviewers for PR {pr_number} in {repo_id}"
                )

            if labels:
                lbl_resp = await client.post(
                    f"/repos/{repo_id}/issues/{pr_number}/labels",
                    json={"labels": list(labels)},
                )
                self._handle_response_errors(
                    lbl_resp, f"add labels to PR {pr_number} in {repo_id}"
                )

            return handle

    async def merge_pull_request(
        self,
        repo_id: str,
        pr_handle: PullRequestHandle,
        strategy: Literal["merge", "squash", "rebase"] = "squash",
    ) -> MergeResult:
        pr_number = pr_handle.pr_number
        async with self._make_client() as client:
            pr_resp = await client.get(f"/repos/{repo_id}/pulls/{pr_number}")
            self._handle_response_errors(pr_resp, f"fetch PR {pr_number} status in {repo_id}")

            data: dict[str, Any] = pr_resp.json()
            mergeable_state = data.get("mergeable_state")
            if mergeable_state == "blocked":
                raise GovernanceError(
                    f"PR {pr_number} merge is blocked by unmet review or status requirements."
                )

            merge_resp = await client.put(
                f"/repos/{repo_id}/pulls/{pr_number}/merge",
                json={"merge_method": strategy},
            )
            self._handle_response_errors(merge_resp, f"merge PR {pr_number} in {repo_id}")

            merge_data: dict[str, Any] = merge_resp.json()
            success: bool = merge_data.get("merged", False)
            sha: str | None = merge_data.get("sha")

            logger.info(
                "pull_request_merged",
                repo=repo_id,
                pr_number=pr_number,
                merge_method=strategy,
            )
            return MergeResult(success=success, merge_commit_sha=sha)

    async def set_branch_protection(
        self,
        repo_id: str,
        branch: str,
        rules: BranchProtectionRules,
    ) -> None:
        raise RecoverableError("set_branch_protection not yet implemented (Phase 2)")

    # ── CI check run operations ───────────────────────────────────────────

    async def get_check_runs(
        self,
        repo_id: str,
        pr_number: int,
    ) -> Sequence[CheckRunInfo]:
        """List CI check runs for the PR's head commit.

        Fetches the PR to resolve its head SHA, then queries the
        check-runs endpoint for that commit.
        """
        async with self._make_client() as client:
            pr_resp = await client.get(f"/repos/{repo_id}/pulls/{pr_number}")
            self._handle_response_errors(pr_resp, f"fetch PR {pr_number} in {repo_id}")
            pr_data: dict[str, Any] = pr_resp.json()
            head_sha: str = str(pr_data.get("head", {}).get("sha", ""))
            if not head_sha:
                raise RecoverableError(
                    f"Could not resolve head SHA for PR {pr_number} in {repo_id}"
                )

            runs_resp = await client.get(
                f"/repos/{repo_id}/commits/{head_sha}/check-runs",
                params={"per_page": "100"},
            )
            self._handle_response_errors(runs_resp, f"check-runs for {head_sha} in {repo_id}")
            runs_data: Any = runs_resp.json()
            raw_runs: list[Any] = cast(list[Any], runs_data.get("check_runs", []))

        result: list[CheckRunInfo] = []
        for run in raw_runs:
            run_dict = cast(dict[str, Any], run) if isinstance(run, dict) else {}
            result.append(
                CheckRunInfo(
                    name=str(run_dict.get("name", "")),
                    status=str(run_dict.get("status", "")),
                    conclusion=run_dict.get("conclusion"),
                    log_url=run_dict.get("details_url"),
                )
            )
        return result

    async def get_check_run_log(self, log_url: str) -> str:
        """Fetch log text from *log_url*.

        Makes an authenticated GET to *log_url*.  Returns the response body as
        a plain string; returns an empty string on HTTP errors so the caller can
        still report the failure without halting.
        """
        async with httpx.AsyncClient(headers=self._make_headers()) as client:
            try:
                response = await client.get(log_url, follow_redirects=True)
                if response.status_code == 200:
                    return response.text
            except httpx.HTTPError:
                pass
        return ""

    async def rerun_failed_checks(self, repo_id: str, pr_number: int) -> None:
        """Re-trigger all failed check runs on the PR.

        Fetches the PR head SHA, then re-requests each check run whose
        ``conclusion`` is ``"failure"`` via the GitHub Checks API.
        """
        async with self._make_client() as client:
            pr_resp = await client.get(f"/repos/{repo_id}/pulls/{pr_number}")
            self._handle_response_errors(
                pr_resp, f"fetch PR {pr_number} for rerun in {repo_id}"
            )
            pr_data: dict[str, Any] = pr_resp.json()
            head_sha: str = str(pr_data.get("head", {}).get("sha", ""))
            if not head_sha:
                raise RecoverableError(
                    f"Could not resolve head SHA for PR {pr_number} in {repo_id}"
                )

            runs_resp = await client.get(
                f"/repos/{repo_id}/commits/{head_sha}/check-runs",
                params={"per_page": "100"},
            )
            self._handle_response_errors(runs_resp, f"check-runs for rerun in {repo_id}")
            runs_data: Any = runs_resp.json()
            raw_runs: list[Any] = cast(list[Any], runs_data.get("check_runs", []))

            rerun_count = 0
            for raw_run in raw_runs:
                rdict = cast(dict[str, Any], raw_run) if isinstance(raw_run, dict) else {}
                if rdict.get("conclusion") != "failure":
                    continue
                run_id: int = int(rdict.get("id", 0))
                rerun_resp = await client.post(
                    f"/repos/{repo_id}/check-runs/{run_id}/rerequest"
                )
                self._handle_response_errors(rerun_resp, f"rerun check {run_id} in {repo_id}")
                rerun_count += 1

        logger.info(
            "ci_checks_rerun",
            repo=repo_id,
            pr_number=pr_number,
            count=rerun_count,
        )
