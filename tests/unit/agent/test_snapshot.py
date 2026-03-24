"""Unit tests for tractable/agent/snapshot.py — TASK-3.1.2.

Covers:
- Snapshot creation produces a valid .tar.gz (AC-1)
- Restore reproduces the exact directory tree (AC-1)
- SHA-256 mismatch raises FatalError (AC-1)
- Cleanup deletes all archives for a given task ID (AC-1 / AC-4)
"""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

import pytest

from tractable.agent.snapshot import cleanup_snapshots, create_snapshot, restore_snapshot
from tractable.errors import FatalError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_working_dir(base: Path) -> Path:
    """Create a small file tree for testing."""
    wd = base / "working_dir"
    wd.mkdir()
    (wd / "main.py").write_text("print('hello')\n")
    (wd / "sub").mkdir()
    (wd / "sub" / "util.py").write_text("def helper(): pass\n")
    return wd


# ---------------------------------------------------------------------------
# Snapshot creation
# ---------------------------------------------------------------------------


def test_create_snapshot_produces_valid_tar_gz() -> None:
    """create_snapshot returns a path to a valid .tar.gz archive."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        wd = _make_working_dir(tmpdir)
        snapshot_dir = tmpdir / "snapshots"

        archive_path, sha256_hash = create_snapshot(wd, snapshot_dir)

        assert Path(archive_path).exists()
        assert archive_path.endswith(".tar.gz")
        assert len(sha256_hash) == 64  # SHA-256 hex = 64 chars
        assert tarfile.is_tarfile(archive_path)


def test_create_snapshot_creates_snapshot_dir_if_missing() -> None:
    """create_snapshot creates snapshot_dir even if it does not yet exist."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        wd = _make_working_dir(tmpdir)
        snapshot_dir = tmpdir / "deep" / "nested" / "snapshots"

        archive_path, _ = create_snapshot(wd, snapshot_dir)

        assert snapshot_dir.exists()
        assert Path(archive_path).parent == snapshot_dir


def test_create_snapshot_archive_contains_working_dir_contents() -> None:
    """The .tar.gz archive contains the files from the working directory."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        wd = _make_working_dir(tmpdir)
        snapshot_dir = tmpdir / "snapshots"

        archive_path, _ = create_snapshot(wd, snapshot_dir)

        with tarfile.open(archive_path, "r:gz") as tar:
            names = tar.getnames()

        # The archive uses "." as the root, so we expect ./main.py etc.
        assert any("main.py" in name for name in names)
        assert any("util.py" in name for name in names)


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


def test_restore_reproduces_exact_directory_tree() -> None:
    """restore_snapshot recreates the exact directory tree from the snapshot."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        wd = _make_working_dir(tmpdir)
        snapshot_dir = tmpdir / "snapshots"

        archive_path, sha256_hash = create_snapshot(wd, snapshot_dir)

        # Corrupt the working directory (simulate partial write).
        (wd / "main.py").write_text("CORRUPTED\n")
        (wd / "extra_file.py").write_text("unexpected\n")

        restore_snapshot(archive_path, sha256_hash, wd)

        assert (wd / "main.py").read_text() == "print('hello')\n"
        assert (wd / "sub" / "util.py").read_text() == "def helper(): pass\n"
        assert not (wd / "extra_file.py").exists()


def test_restore_raises_fatal_error_on_hash_mismatch() -> None:
    """restore_snapshot raises FatalError when the SHA-256 does not match."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        wd = _make_working_dir(tmpdir)
        snapshot_dir = tmpdir / "snapshots"

        archive_path, _ = create_snapshot(wd, snapshot_dir)
        wrong_hash = "a" * 64  # clearly wrong hash

        with pytest.raises(FatalError, match="integrity check failed"):
            restore_snapshot(archive_path, wrong_hash, wd)


def test_restore_does_not_modify_working_dir_on_hash_mismatch() -> None:
    """Working directory is NOT modified when hash mismatch raises FatalError."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        wd = _make_working_dir(tmpdir)
        snapshot_dir = tmpdir / "snapshots"

        archive_path, _ = create_snapshot(wd, snapshot_dir)

        # Write a known marker to working_dir.
        (wd / "main.py").write_text("MARKER\n")

        with pytest.raises(FatalError):
            restore_snapshot(archive_path, "b" * 64, wd)

        # Working dir should still have the marker — restore was not applied.
        assert (wd / "main.py").read_text() == "MARKER\n"


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def test_cleanup_snapshots_deletes_all_tar_gz_for_task() -> None:
    """cleanup_snapshots deletes all .tar.gz archives in snapshot_dir."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        wd = _make_working_dir(tmpdir)
        snapshot_dir = tmpdir / "snapshots"

        # Create multiple snapshots for the same task.
        archive_path1, _ = create_snapshot(wd, snapshot_dir)
        archive_path2, _ = create_snapshot(wd, snapshot_dir)

        assert Path(archive_path1).exists()
        assert Path(archive_path2).exists()

        cleanup_snapshots(snapshot_dir, task_id="task-123")

        assert not Path(archive_path1).exists()
        assert not Path(archive_path2).exists()


def test_cleanup_snapshots_noop_when_dir_missing() -> None:
    """cleanup_snapshots does not raise if snapshot_dir does not exist."""
    with tempfile.TemporaryDirectory() as tmp:
        nonexistent = Path(tmp) / "does_not_exist"
        # Should not raise.
        cleanup_snapshots(nonexistent, task_id="task-xyz")


def test_cleanup_snapshots_noop_when_no_archives_present() -> None:
    """cleanup_snapshots handles an empty snapshot directory gracefully."""
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_dir = Path(tmp) / "snapshots"
        snapshot_dir.mkdir()
        # Should not raise.
        cleanup_snapshots(snapshot_dir, task_id="task-xyz")


def test_snapshot_cleanup_on_completion() -> None:
    """No .tar.gz files remain in snapshot_dir after cleanup (AC-4).

    Simulates the COORDINATING node calling cleanup_snapshots once a task
    reaches COMPLETED status.  After cleanup, the snapshot_dir contains no
    ``.tar.gz`` files for the completed task.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        wd = _make_working_dir(tmpdir)
        task_id = "task-completed-123"
        snapshot_dir = tmpdir / "tractable_snapshots" / task_id

        # Simulate multiple checkpoint transitions creating snapshots.
        archive1, _ = create_snapshot(wd, snapshot_dir)
        archive2, _ = create_snapshot(wd, snapshot_dir)
        archive3, _ = create_snapshot(wd, snapshot_dir)

        for path in (archive1, archive2, archive3):
            assert Path(path).exists(), f"Expected archive to exist: {path}"

        # Simulate COORDINATING node cleanup step.
        cleanup_snapshots(snapshot_dir, task_id=task_id)

        # No .tar.gz files should remain.
        remaining = list(snapshot_dir.glob("*.tar.gz"))
        assert remaining == [], (
            f"Expected no .tar.gz files after cleanup, found: {remaining}"
        )
