"""Working-directory snapshot helpers for crash-safe checkpoint restore.

TASK-3.1.2: Creates and restores `.tar.gz` archives of an agent's working
directory so that a crashed agent can return the filesystem to a known-clean
state before re-entering the EXECUTING node.

Usage
-----
- ``create_snapshot`` is called by the snapshot wrapper in ``workflow.py``
  after every node completes successfully.
- ``restore_snapshot`` is called by ``resume_task()`` before dispatching to
  the LangGraph node, reverting any partial writes made during the crashed run.
- ``cleanup_snapshots`` is called from the COORDINATING node's cleanup step
  once a task reaches COMPLETED or FAILED.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
import uuid
from pathlib import Path

import structlog

from tractable.errors import FatalError

_log = structlog.get_logger()


def create_snapshot(working_dir: Path, snapshot_dir: Path) -> tuple[str, str]:
    """Create a `.tar.gz` archive of *working_dir* inside *snapshot_dir*.

    Parameters
    ----------
    working_dir:
        Directory whose contents should be archived.
    snapshot_dir:
        Destination directory for the archive file.  Created if it does not
        exist.

    Returns
    -------
    tuple[str, str]
        ``(archive_path, sha256_hash)`` where *archive_path* is the absolute
        path to the newly created ``.tar.gz`` file and *sha256_hash* is its
        SHA-256 hex digest.
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"snapshot_{uuid.uuid4().hex}.tar.gz"
    archive_path = snapshot_dir / archive_name

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(working_dir, arcname=".")

    sha256_hash = _sha256(archive_path)

    _log.debug(
        "snapshot_created",
        archive=str(archive_path),
        sha256=sha256_hash,
    )
    return str(archive_path), sha256_hash


def restore_snapshot(
    snapshot_path: str,
    expected_hash: str,
    working_dir: Path,
) -> None:
    """Restore a working directory from a previously created snapshot.

    Verifies the SHA-256 integrity of the archive before extracting.
    Clears *working_dir* before extraction so the restore is idempotent.

    Parameters
    ----------
    snapshot_path:
        Absolute path to the ``.tar.gz`` archive.
    expected_hash:
        SHA-256 hex digest the archive must match.  If it does not match,
        ``FatalError`` is raised and the working directory is **not** modified.
    working_dir:
        Directory to restore into.  Existing contents are replaced.

    Raises
    ------
    FatalError
        If the on-disk SHA-256 of the archive does not match *expected_hash*
        (indicating corruption).
    """
    archive = Path(snapshot_path)
    actual_hash = _sha256(archive)

    if actual_hash != expected_hash:
        raise FatalError(
            f"Snapshot integrity check failed for {snapshot_path!r}: "
            f"expected {expected_hash!r}, got {actual_hash!r}."
        )

    # Clear the working directory before restoring so stale partial writes are
    # removed.  We extract into a temp sibling directory first, then swap, to
    # avoid leaving working_dir empty if extraction fails.
    parent = working_dir.parent
    tmp_restore = Path(tempfile.mkdtemp(dir=parent, prefix="_tractable_restore_"))
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp_restore)  # noqa: S202  # controlled archive, no untrusted input

        # Remove old working_dir content then move the restored tree in.
        if working_dir.exists():
            shutil.rmtree(working_dir)
        shutil.move(str(tmp_restore), str(working_dir))
    except Exception:
        # If anything goes wrong, clean up the temp dir and re-raise.
        if tmp_restore.exists():
            shutil.rmtree(tmp_restore, ignore_errors=True)
        raise

    _log.debug(
        "snapshot_restored",
        archive=snapshot_path,
        working_dir=str(working_dir),
    )


def cleanup_snapshots(snapshot_dir: Path, task_id: str) -> None:
    """Delete all snapshot archives for *task_id* from *snapshot_dir*.

    Safe to call even if *snapshot_dir* does not exist or is already empty.

    Parameters
    ----------
    snapshot_dir:
        Directory that was used as the ``snapshot_dir`` when calling
        ``create_snapshot``.
    task_id:
        Identifier of the task whose snapshots should be removed.  All
        ``snapshot_*.tar.gz`` files inside ``snapshot_dir / task_id`` (or
        directly in ``snapshot_dir`` if organised that way) are deleted.
        This implementation treats *snapshot_dir* as the per-task directory
        (i.e. ``Path(tempfile.gettempdir()) / "tractable_snapshots" / task_id``)
        and removes all ``.tar.gz`` files within it.
    """
    if not snapshot_dir.exists():
        return

    removed = 0
    for archive in snapshot_dir.glob("*.tar.gz"):
        try:
            archive.unlink()
            removed += 1
        except OSError:
            _log.warning(
                "snapshot_cleanup_failed",
                archive=str(archive),
                task_id=task_id,
            )

    _log.debug("snapshots_cleaned", task_id=task_id, removed=removed)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of the file at *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
