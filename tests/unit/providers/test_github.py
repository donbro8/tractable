"""Unit tests for GitHubProvider.

All HTTP calls are intercepted by respx — no real network traffic occurs.
Git subprocess calls are mocked with unittest.mock.

Integration tests that require a real GitHub token are tagged
@pytest.mark.integration and skipped in unit test runs.
"""

from __future__ import annotations

import base64
import pathlib
from collections.abc import Sequence
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx
import structlog.testing

from tractable.errors import FatalError, GovernanceError, RecoverableError, TransientError
from tractable.protocols.git_provider import GitProvider
from tractable.providers.factory import create_git_provider
from tractable.providers.github import GitHubProvider
from tractable.types.config import GitProviderConfig
from tractable.types.git import FileEntry

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def github_config() -> GitProviderConfig:
    return GitProviderConfig(
        provider_type="github",
        credentials_secret_ref="TEST_GITHUB_TOKEN",
    )


@pytest.fixture
def provider(
    monkeypatch: pytest.MonkeyPatch,
    github_config: GitProviderConfig,
) -> GitHubProvider:
    monkeypatch.setenv("TEST_GITHUB_TOKEN", "ghp_testtoken1234567890")
    return GitHubProvider(github_config)


# ── Protocol conformance ──────────────────────────────────────────────────────


def test_github_provider_satisfies_protocol(
    monkeypatch: pytest.MonkeyPatch,
    github_config: GitProviderConfig,
) -> None:
    """GitHubProvider is an instance of the GitProvider Protocol."""
    monkeypatch.setenv("TEST_GITHUB_TOKEN", "ghp_testtoken1234567890")
    p = GitHubProvider(github_config)
    assert isinstance(p, GitProvider)


# ── Constructor credential validation ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_credentials_raises_at_first_call(
    monkeypatch: pytest.MonkeyPatch,
    github_config: GitProviderConfig,
    tmp_path: pathlib.Path,
) -> None:
    """A missing credentials env var raises FatalError at the first API call."""
    monkeypatch.delenv("TEST_GITHUB_TOKEN", raising=False)
    p = GitHubProvider(github_config)  # construction succeeds — resolution deferred
    with pytest.raises(FatalError, match="TEST_GITHUB_TOKEN"), patch("subprocess.run"):
        await p.clone("owner/repo", str(tmp_path))


def test_default_constructor_reads_github_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """GitHubProvider() with no args constructs successfully; token resolved at call time."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_default_token_xyz")
    p = GitHubProvider()
    assert isinstance(p, GitProvider)


@pytest.mark.asyncio
async def test_default_constructor_raises_when_github_token_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Missing GITHUB_TOKEN raises FatalError at the first credential-needing call."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    p = GitHubProvider()  # construction succeeds — resolution deferred
    with pytest.raises(FatalError, match="GITHUB_TOKEN"), patch("subprocess.run"):
        await p.clone("owner/repo", str(tmp_path))


# ── clone ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_clone_success(
    provider: GitHubProvider,
    tmp_path: pathlib.Path,
) -> None:
    """Successful clone returns the target path."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = await provider.clone("owner/repo", str(tmp_path), branch="main")

    assert result == str(tmp_path)
    mock_run.assert_called_once()
    call_args: list[str] = mock_run.call_args[0][0]
    assert "git" in call_args
    assert "clone" in call_args
    assert "--branch" in call_args
    assert "main" in call_args


@pytest.mark.asyncio
async def test_clone_sparse_checkout(
    provider: GitHubProvider,
    tmp_path: pathlib.Path,
) -> None:
    """Sparse checkout issues two subprocess calls."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        await provider.clone(
            "owner/repo",
            str(tmp_path),
            branch="main",
            sparse_paths=["src/", "tests/"],
        )

    assert mock_run.call_count == 2
    sparse_call: list[str] = mock_run.call_args_list[1][0][0]
    assert "sparse-checkout" in sparse_call
    assert "src/" in sparse_call
    assert "tests/" in sparse_call


@pytest.mark.asyncio
async def test_clone_failure_raises_recoverable_error(
    provider: GitHubProvider,
    tmp_path: pathlib.Path,
) -> None:
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=128, stderr="not found")
        with pytest.raises(TransientError, match="owner/repo"):
            await provider.clone("owner/repo", str(tmp_path))


@pytest.mark.asyncio
async def test_clone_token_not_in_log_output(
    provider: GitHubProvider,
    tmp_path: pathlib.Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """The GitHub token must never appear in structlog output."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        await provider.clone("owner/repo", str(tmp_path))

    out, err = capfd.readouterr()
    assert "ghp_testtoken1234567890" not in out
    assert "ghp_testtoken1234567890" not in err


# ── clone: URL validation and credential injection (TASK-2.2.2) ──────────────


@pytest.mark.asyncio
async def test_clone_full_url_credentials_not_in_subprocess_url(
    provider: GitHubProvider,
    tmp_path: pathlib.Path,
) -> None:
    """Full HTTPS URL → token NOT in subprocess args; credentials via env; token not in logs."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        with structlog.testing.capture_logs() as cap_logs:
            await provider.clone("https://github.com/org/repo.git", str(tmp_path))

    call_args: list[str] = mock_run.call_args[0][0]
    # Token must NOT appear in any subprocess argument (including the URL).
    for arg in call_args:
        assert "ghp_testtoken1234567890" not in str(arg)
    # Clone URL must be plain HTTPS without embedded credentials.
    clone_url = next((a for a in call_args if "github.com" in a), "")
    assert "x-access-token:" not in clone_url
    # Credentials must be passed via env, not URL.
    call_kwargs = mock_run.call_args[1]
    env_passed: dict[str, str] = call_kwargs.get("env", {})
    assert "GIT_CONFIG_VALUE_0" in env_passed

    # Token must not appear in any structlog output.
    for entry in cap_logs:
        for v in entry.values():
            assert "ghp_testtoken1234567890" not in str(v)


@pytest.mark.asyncio
async def test_clone_shell_metacharacters_raise_governance_error(
    provider: GitHubProvider,
    tmp_path: pathlib.Path,
) -> None:
    """AC-2: URL with shell metacharacters raises GovernanceError before subprocess."""
    with (
        patch("subprocess.run") as mock_run,
        pytest.raises(GovernanceError, match="invalid characters"),
    ):
        await provider.clone("https://github.com/org/repo.git; rm -rf /", str(tmp_path))
    mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_clone_http_scheme_raises_governance_error(
    provider: GitHubProvider,
    tmp_path: pathlib.Path,
) -> None:
    """AC-3: HTTP (non-HTTPS) URL raises GovernanceError."""
    with (
        patch("subprocess.run") as mock_run,
        pytest.raises(GovernanceError, match="scheme must be 'https'"),
    ):
        await provider.clone("http://github.com/org/repo.git", str(tmp_path))
    mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_clone_disallowed_host_raises_governance_error(
    provider: GitHubProvider,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-4: Host not in allowlist raises GovernanceError."""
    monkeypatch.delenv("TRACTABLE_ALLOWED_GIT_HOSTS", raising=False)
    with (
        patch("subprocess.run") as mock_run,
        pytest.raises(GovernanceError, match="not in the allowed list"),
    ):
        await provider.clone("https://evil.example.com/org/repo.git", str(tmp_path))
    mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_clone_cleanup_on_post_clone_fn_failure(
    provider: GitHubProvider,
    tmp_path: pathlib.Path,
) -> None:
    """AC-5: Successful clone + post_clone_fn raises → target directory deleted."""

    def parse_file(path: str) -> None:
        raise RuntimeError("parse_file failed")

    clone_dir = tmp_path / "cloned"
    clone_dir.mkdir()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        with pytest.raises(RuntimeError, match="parse_file failed"):
            await provider.clone(
                "owner/repo",
                str(clone_dir),
                post_clone_fn=parse_file,
            )

    assert not clone_dir.exists(), "Cloned directory must be deleted on post-clone failure"


@pytest.mark.asyncio
async def test_clone_cleanup_on_clone_failure(
    provider: GitHubProvider,
    tmp_path: pathlib.Path,
) -> None:
    """Partial clone dir is removed when subprocess returns non-zero."""
    clone_dir = tmp_path / "partial"
    clone_dir.mkdir()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="auth fail")
        with pytest.raises(TransientError):
            await provider.clone("owner/repo", str(clone_dir))

    assert not clone_dir.exists()


@pytest.mark.asyncio
async def test_clone_shorthand_id_still_works(
    provider: GitHubProvider,
    tmp_path: pathlib.Path,
) -> None:
    """Shorthand 'org/repo' format continues to work after URL validation added."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = await provider.clone("owner/repo", str(tmp_path))
    assert result == str(tmp_path)


# ── get_file_content ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_get_file_content_success(provider: GitHubProvider) -> None:
    """Returns base64-decoded bytes for a valid file."""
    encoded = base64.b64encode(b"# Hello World\n").decode()
    respx.get("https://api.github.com/repos/owner/repo/contents/README.md").mock(
        return_value=httpx.Response(200, json={"content": encoded, "encoding": "base64"})
    )

    content = await provider.get_file_content("owner/repo", "README.md", ref="main")
    assert content == b"# Hello World\n"


@pytest.mark.asyncio
@respx.mock
async def test_get_file_content_rate_limit_raises_transient_error(
    provider: GitHubProvider,
) -> None:
    """GitHub 403 with Retry-After header raises TransientError with correct retry_after."""
    respx.get("https://api.github.com/repos/owner/repo/contents/README.md").mock(
        return_value=httpx.Response(
            403,
            headers={"Retry-After": "30"},
            json={"message": "API rate limit exceeded"},
        )
    )

    with pytest.raises(TransientError) as exc_info:
        await provider.get_file_content("owner/repo", "README.md")

    assert exc_info.value.retry_after == 30


@pytest.mark.asyncio
@respx.mock
async def test_get_file_content_not_found_raises_recoverable_error(
    provider: GitHubProvider,
) -> None:
    """GitHub 404 raises RecoverableError mentioning the resource path."""
    respx.get("https://api.github.com/repos/owner/repo/contents/missing.py").mock(
        return_value=httpx.Response(404, json={"message": "Not Found"})
    )

    with pytest.raises(RecoverableError, match="missing.py"):
        await provider.get_file_content("owner/repo", "missing.py")


@pytest.mark.asyncio
@respx.mock
async def test_get_file_content_recoverable_on_404(
    provider: GitHubProvider,
) -> None:
    """AC-3: get_file_content with 404 response → RecoverableError (named for AC verification)."""
    respx.get("https://api.github.com/repos/owner/repo/contents/src/app.py").mock(
        return_value=httpx.Response(404, json={"message": "Not Found"})
    )

    with pytest.raises(RecoverableError):
        await provider.get_file_content("owner/repo", "src/app.py")


# ── list_files ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_list_files_success(provider: GitHubProvider) -> None:
    """Returns a sequence of FileEntry objects for a directory listing."""
    respx.get("https://api.github.com/repos/owner/repo/contents/src").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"path": "src/main.py", "type": "file", "size": 1024, "sha": "abc123"},
                {"path": "src/utils", "type": "dir", "size": 0, "sha": "def456"},
            ],
        )
    )

    files: Sequence[FileEntry] = await provider.list_files("owner/repo", "src", ref="main")

    assert len(files) == 2
    assert files[0].path == "src/main.py"
    assert files[0].is_directory is False
    assert files[0].size_bytes == 1024
    assert files[1].path == "src/utils"
    assert files[1].is_directory is True


@pytest.mark.asyncio
@respx.mock
async def test_list_files_not_found_raises_recoverable_error(
    provider: GitHubProvider,
) -> None:
    respx.get("https://api.github.com/repos/owner/repo/contents/nonexistent").mock(
        return_value=httpx.Response(404, json={"message": "Not Found"})
    )

    with pytest.raises(RecoverableError):
        await provider.list_files("owner/repo", "nonexistent")


# ── get_diff ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_get_diff_success(provider: GitHubProvider) -> None:
    """Returns the raw unified diff string."""
    diff_text = "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n"
    respx.get("https://api.github.com/repos/owner/repo/compare/main...feature").mock(
        return_value=httpx.Response(200, text=diff_text)
    )

    diff = await provider.get_diff("owner/repo", "main", "feature")
    assert diff == diff_text


# ── get_commit_history ────────────────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_get_commit_history_success(provider: GitHubProvider) -> None:
    """Returns a sequence of CommitEntry objects."""
    respx.get("https://api.github.com/repos/owner/repo/commits").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "sha": "abc123def456",
                    "commit": {
                        "message": "fix: update auth middleware",
                        "author": {"name": "Alice", "date": "2026-03-17T10:00:00Z"},
                    },
                    "files": [{"filename": "src/auth.py"}],
                }
            ],
        )
    )

    history = await provider.get_commit_history("owner/repo", limit=10)

    assert len(history) == 1
    assert history[0].sha == "abc123def456"
    assert history[0].author == "Alice"
    assert history[0].files_changed == ["src/auth.py"]
    assert history[0].timestamp == datetime(2026, 3, 17, 10, 0, 0, tzinfo=UTC)


@pytest.mark.asyncio
@respx.mock
async def test_get_commit_history_with_path_filter(provider: GitHubProvider) -> None:
    """The path filter is forwarded as a query parameter."""
    route = respx.get("https://api.github.com/repos/owner/repo/commits").mock(
        return_value=httpx.Response(200, json=[])
    )

    await provider.get_commit_history("owner/repo", path="src/auth.py", limit=5)

    assert route.called
    request = route.calls.last.request
    url_str = str(request.url)
    assert "path=src" in url_str
    assert "per_page=5" in url_str


# ── Write operations ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_create_branch_success(provider: GitHubProvider) -> None:
    """Returns the SHA of the newly created branch head."""
    respx.get("https://api.github.com/repos/owner/repo/git/refs/heads/feature/x").mock(
        return_value=httpx.Response(404)
    )
    respx.get("https://api.github.com/repos/owner/repo/git/refs/heads/main").mock(
        return_value=httpx.Response(200, json={"object": {"sha": "abc123fromsha"}})
    )
    respx.post("https://api.github.com/repos/owner/repo/git/refs").mock(
        return_value=httpx.Response(201, json={"object": {"sha": "abc123fromsha"}})
    )

    sha = await provider.create_branch("owner/repo", "feature/x", from_ref="main")
    assert sha == "abc123fromsha"


@pytest.mark.asyncio
@respx.mock
async def test_create_branch_already_exists(provider: GitHubProvider) -> None:
    respx.get("https://api.github.com/repos/owner/repo/git/refs/heads/feature/x").mock(
        return_value=httpx.Response(200)
    )
    with pytest.raises(RecoverableError, match="Branch already exists"):
        await provider.create_branch("owner/repo", "feature/x")


@pytest.mark.asyncio
@respx.mock
async def test_create_pull_request_success(provider: GitHubProvider) -> None:
    respx.post("https://api.github.com/repos/owner/repo/pulls").mock(
        return_value=httpx.Response(
            201, json={"number": 42, "html_url": "https://github.com/owner/repo/pull/42"}
        )
    )
    respx.post("https://api.github.com/repos/owner/repo/pulls/42/requested_reviewers").mock(
        return_value=httpx.Response(201)
    )
    respx.post("https://api.github.com/repos/owner/repo/issues/42/labels").mock(
        return_value=httpx.Response(200)
    )

    handle = await provider.create_pull_request(
        "owner/repo", "Title", "Body", "feature/x", "main", ["alice"], ["bug"]
    )
    assert handle.pr_number == 42
    assert handle.url == "https://github.com/owner/repo/pull/42"
    assert handle.head_branch == "feature/x"
    assert handle.base_branch == "main"


@pytest.mark.asyncio
@respx.mock
async def test_merge_pull_request_success(provider: GitHubProvider) -> None:
    from tractable.types.git import PullRequestHandle

    pr = PullRequestHandle(
        provider="github",
        repo_id="owner/repo",
        pr_number=42,
        url="https://github.com/owner/repo/pull/42",
        head_branch="feature/x",
        base_branch="main",
    )
    respx.get("https://api.github.com/repos/owner/repo/pulls/42").mock(
        return_value=httpx.Response(200, json={"mergeable_state": "clean"})
    )
    respx.put("https://api.github.com/repos/owner/repo/pulls/42/merge").mock(
        return_value=httpx.Response(200, json={"merged": True, "sha": "mergedsha456"})
    )

    result = await provider.merge_pull_request("owner/repo", pr, strategy="squash")
    assert result.success is True
    assert result.merge_commit_sha == "mergedsha456"


@pytest.mark.asyncio
@respx.mock
async def test_merge_pull_request_blocked(provider: GitHubProvider) -> None:
    from tractable.types.git import PullRequestHandle

    pr = PullRequestHandle(
        provider="github",
        repo_id="owner/repo",
        pr_number=42,
        url="https://github.com/owner/repo/pull/42",
        head_branch="feature/x",
        base_branch="main",
    )
    respx.get("https://api.github.com/repos/owner/repo/pulls/42").mock(
        return_value=httpx.Response(200, json={"mergeable_state": "blocked"})
    )

    with pytest.raises(GovernanceError, match="blocked by unmet review"):
        await provider.merge_pull_request("owner/repo", pr)


@pytest.mark.asyncio
@respx.mock
async def test_create_branch_rate_limit_raises_transient_error(
    provider: GitHubProvider,
) -> None:
    """create_branch 429 raises TransientError with retry_after from header."""
    respx.get("https://api.github.com/repos/owner/repo/git/refs/heads/feature/x").mock(
        return_value=httpx.Response(
            429,
            headers={"Retry-After": "20"},
            json={"message": "rate limit"},
        )
    )
    with pytest.raises(TransientError) as exc_info:
        await provider.create_branch("owner/repo", "feature/x")
    assert exc_info.value.retry_after == 20


@pytest.mark.asyncio
@respx.mock
async def test_create_pull_request_rate_limit_raises_transient_error(
    provider: GitHubProvider,
) -> None:
    """create_pull_request 429 raises TransientError with retry_after from header."""
    respx.post("https://api.github.com/repos/owner/repo/pulls").mock(
        return_value=httpx.Response(
            429,
            headers={"Retry-After": "35"},
            json={"message": "rate limit"},
        )
    )
    with pytest.raises(TransientError) as exc_info:
        await provider.create_pull_request("owner/repo", "Title", "Body", "feature/x")
    assert exc_info.value.retry_after == 35


@pytest.mark.asyncio
@respx.mock
async def test_merge_pull_request_rate_limit_raises_transient_error(
    provider: GitHubProvider,
) -> None:
    """merge_pull_request 429 (on PR fetch) raises TransientError with retry_after."""
    from tractable.types.git import PullRequestHandle

    pr = PullRequestHandle(
        provider="github",
        repo_id="owner/repo",
        pr_number=42,
        url="https://github.com/owner/repo/pull/42",
        head_branch="feature/x",
        base_branch="main",
    )
    respx.get("https://api.github.com/repos/owner/repo/pulls/42").mock(
        return_value=httpx.Response(
            429,
            headers={"Retry-After": "10"},
            json={"message": "rate limit"},
        )
    )
    with pytest.raises(TransientError) as exc_info:
        await provider.merge_pull_request("owner/repo", pr)
    assert exc_info.value.retry_after == 10


# ── Structlog output tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_create_branch_logs_info(provider: GitHubProvider) -> None:
    """create_branch emits a structlog entry at info level with repo and branch_name."""
    respx.get("https://api.github.com/repos/owner/repo/git/refs/heads/feature/x").mock(
        return_value=httpx.Response(404)
    )
    respx.get("https://api.github.com/repos/owner/repo/git/refs/heads/main").mock(
        return_value=httpx.Response(200, json={"object": {"sha": "deadbeef"}})
    )
    respx.post("https://api.github.com/repos/owner/repo/git/refs").mock(
        return_value=httpx.Response(201, json={"object": {"sha": "deadbeef"}})
    )

    with structlog.testing.capture_logs() as cap_logs:
        await provider.create_branch("owner/repo", "feature/x", from_ref="main")

    info_logs = [
        e for e in cap_logs if e.get("log_level") == "info" and e.get("event") == "branch_created"
    ]
    assert info_logs, f"Expected branch_created info log; got {cap_logs}"
    entry = info_logs[0]
    assert entry["repo"] == "owner/repo"
    assert entry["branch_name"] == "feature/x"


@pytest.mark.asyncio
@respx.mock
async def test_create_pull_request_logs_info(provider: GitHubProvider) -> None:
    """create_pull_request emits pull_request_created at info with required fields."""
    respx.post("https://api.github.com/repos/owner/repo/pulls").mock(
        return_value=httpx.Response(
            201,
            json={"number": 7, "html_url": "https://github.com/owner/repo/pull/7"},
        )
    )

    with structlog.testing.capture_logs() as cap_logs:
        await provider.create_pull_request(
            "owner/repo", "Fix bug", "Body text", "feature/fix", "main"
        )

    info_logs = [
        e
        for e in cap_logs
        if e.get("log_level") == "info" and e.get("event") == "pull_request_created"
    ]
    assert info_logs, f"Expected pull_request_created info log; got {cap_logs}"
    entry = info_logs[0]
    assert entry["repo"] == "owner/repo"
    assert entry["pr_number"] == 7
    assert entry["head_branch"] == "feature/fix"
    assert entry["base_branch"] == "main"


@pytest.mark.asyncio
@respx.mock
async def test_merge_pull_request_logs_info(provider: GitHubProvider) -> None:
    """merge_pull_request emits pull_request_merged at info with required fields."""
    from tractable.types.git import PullRequestHandle

    pr = PullRequestHandle(
        provider="github",
        repo_id="owner/repo",
        pr_number=99,
        url="https://github.com/owner/repo/pull/99",
        head_branch="feature/y",
        base_branch="main",
    )
    respx.get("https://api.github.com/repos/owner/repo/pulls/99").mock(
        return_value=httpx.Response(200, json={"mergeable_state": "clean"})
    )
    respx.put("https://api.github.com/repos/owner/repo/pulls/99/merge").mock(
        return_value=httpx.Response(200, json={"merged": True, "sha": "abc000"})
    )

    with structlog.testing.capture_logs() as cap_logs:
        await provider.merge_pull_request("owner/repo", pr, strategy="squash")

    info_logs = [
        e
        for e in cap_logs
        if e.get("log_level") == "info" and e.get("event") == "pull_request_merged"
    ]
    assert info_logs, f"Expected pull_request_merged info log; got {cap_logs}"
    entry = info_logs[0]
    assert entry["repo"] == "owner/repo"
    assert entry["pr_number"] == 99
    assert entry["merge_method"] == "squash"


@pytest.mark.asyncio
async def test_set_branch_protection_not_implemented(provider: GitHubProvider) -> None:
    from tractable.types.git import BranchProtectionRules

    with pytest.raises(RecoverableError, match="Phase 2"):
        await provider.set_branch_protection("owner/repo", "main", BranchProtectionRules())


# ── Factory ───────────────────────────────────────────────────────────────────


def test_factory_creates_github_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_GITHUB_TOKEN", "ghp_factory_token")
    config = GitProviderConfig(
        provider_type="github",
        credentials_secret_ref="TEST_GITHUB_TOKEN",
    )
    p = create_git_provider(config)
    assert isinstance(p, GitProvider)
    assert isinstance(p, GitHubProvider)


def test_factory_raises_for_unsupported_provider() -> None:
    config = GitProviderConfig(
        provider_type="gitlab",
        credentials_secret_ref="GITLAB_TOKEN",
    )
    with pytest.raises(RecoverableError, match="gitlab"):
        create_git_provider(config)


# ── HTTP error mapping ────────────────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_handle_response_errors_401_fatal(provider: GitHubProvider) -> None:
    """GitHub 401 (invalid credentials) raises FatalError."""
    respx.get("https://api.github.com/repos/owner/repo/contents/README.md").mock(
        return_value=httpx.Response(401, json={"message": "Bad credentials"})
    )

    with pytest.raises(FatalError, match="authentication failed"):
        await provider.get_file_content("owner/repo", "README.md")


@pytest.mark.asyncio
@respx.mock
async def test_handle_response_errors_429_transient(provider: GitHubProvider) -> None:
    """GitHub 429 (rate limit) raises TransientError with retry_after."""
    respx.get("https://api.github.com/repos/owner/repo/contents/README.md").mock(
        return_value=httpx.Response(
            429,
            headers={"Retry-After": "45"},
            json={"message": "rate limit"},
        )
    )

    with pytest.raises(TransientError) as exc_info:
        await provider.get_file_content("owner/repo", "README.md")

    assert exc_info.value.retry_after == 45


@pytest.mark.asyncio
@respx.mock
async def test_handle_response_errors_500_transient(provider: GitHubProvider) -> None:
    """GitHub 500 (server error) raises TransientError."""
    respx.get("https://api.github.com/repos/owner/repo/contents/README.md").mock(
        return_value=httpx.Response(500, json={"message": "Internal Server Error"})
    )

    with pytest.raises(TransientError, match="server error 500"):
        await provider.get_file_content("owner/repo", "README.md")


@pytest.mark.asyncio
@respx.mock
async def test_handle_response_errors_503_transient(provider: GitHubProvider) -> None:
    """GitHub 503 (service unavailable) raises TransientError."""
    respx.get("https://api.github.com/repos/owner/repo/contents/README.md").mock(
        return_value=httpx.Response(503, json={"message": "Service Unavailable"})
    )

    with pytest.raises(TransientError, match="server error 503"):
        await provider.get_file_content("owner/repo", "README.md")


@pytest.mark.asyncio
async def test_clone_transient_error_on_nonzero_exit(
    provider: GitHubProvider,
    tmp_path: pathlib.Path,
) -> None:
    """AC-2: subprocess exit code 128 (git clone failure) raises TransientError."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=128, stderr="fatal: not found")
        with pytest.raises(TransientError, match="git exited 128"):
            await provider.clone("owner/repo", str(tmp_path))


# ── Credential hygiene (TASK-3.3.1) ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_token_not_in_log_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """AC-1: Token never appears in any structlog output during clone()."""
    monkeypatch.setenv("GITHUB_BOT_TOKEN", "ghp_fakefakefake")
    config = GitProviderConfig(
        provider_type="github",
        credentials_secret_ref="GITHUB_BOT_TOKEN",
    )
    provider = GitHubProvider(config)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        with structlog.testing.capture_logs() as cap_logs:
            await provider.clone("owner/repo", str(tmp_path), branch="main")

    for entry in cap_logs:
        for v in entry.values():
            assert "ghp_fakefakefake" not in str(v)


@pytest.mark.asyncio
async def test_token_not_in_subprocess_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """AC-2: Token never appears in any subprocess.run argument during clone()."""
    monkeypatch.setenv("GITHUB_BOT_TOKEN", "ghp_fakefakefake")
    config = GitProviderConfig(
        provider_type="github",
        credentials_secret_ref="GITHUB_BOT_TOKEN",
    )
    provider = GitHubProvider(config)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        await provider.clone("owner/repo", str(tmp_path), branch="main")

    call_args: list[str] = mock_run.call_args[0][0]
    for arg in call_args:
        assert "ghp_fakefakefake" not in str(arg), (
            f"Token found in subprocess arg: {arg!r}"
        )


# ── Integration tests (skipped in unit test runs) ─────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_clone_public_repo(tmp_path: pathlib.Path) -> None:
    """Clone a real public GitHub repo and verify files exist."""
    import os

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        pytest.skip("GITHUB_TOKEN not set")

    config = GitProviderConfig(
        provider_type="github",
        credentials_secret_ref="GITHUB_TOKEN",
    )
    p = GitHubProvider(config)
    result = await p.clone(
        "octocat/Hello-World",
        str(tmp_path / "hello-world"),
        branch="master",
    )
    assert result == str(tmp_path / "hello-world")
    assert pathlib.Path(result).exists()
    assert any(pathlib.Path(result).iterdir())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_get_file_content(tmp_path: pathlib.Path) -> None:  # noqa: ARG001
    """Fetch README from a real public GitHub repo."""
    import os

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        pytest.skip("GITHUB_TOKEN not set")

    config = GitProviderConfig(
        provider_type="github",
        credentials_secret_ref="GITHUB_TOKEN",
    )
    p = GitHubProvider(config)
    content = await p.get_file_content("octocat/Hello-World", "README", ref="master")
    assert len(content) > 0
