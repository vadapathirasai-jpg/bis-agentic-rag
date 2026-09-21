"""Command-line entry point and scheduler runner for BIS Sahayak incremental source synchronization.

Provides a safe, idempotent, and standalone execution command suitable for periodic
invocation via Windows Task Scheduler or manual terminal commands.

Features:
- Windows-friendly local file locking (prevents concurrent sync processes)
- Supports --dry-run for planning without mutations
- Structured Python logging throughout execution
- Human-readable summary output upon completion
- Non-zero exit code on failure
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Generator, Optional

# Ensure backend directory is in sys.path
CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE.parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.source_synchronizer import (
    DEFAULT_COLLECTION_NAME,
    SourceSynchronizer,
    SyncSafetyError,
)

logger = logging.getLogger("bis_sahayak.sync")


class SyncLockedError(Exception):
    """Raised when another synchronization process currently holds the lock."""


@contextmanager
def acquire_sync_lock(lock_path: Path) -> Generator[bool, None, None]:
    """Acquire an exclusive, non-blocking file lock suitable for Windows.

    Uses msvcrt.locking on Windows (with fcntl fallback on POSIX systems).
    If another process holds the lock, raises SyncLockedError.

    Args:
        lock_path: Path to the lock file.

    Yields:
        True if lock was successfully acquired.

    Raises:
        SyncLockedError: If the lock file is currently locked by another process.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = None
    file_descriptor = None

    try:
        # Open in read-write / create binary mode
        lock_file = open(lock_path, "a+b")
        file_descriptor = lock_file.fileno()

        # Non-blocking lock attempt
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0)
            try:
                msvcrt.locking(file_descriptor, msvcrt.LK_NBLCK, 1)
            except (OSError, PermissionError) as err:
                raise SyncLockedError(
                    f"Lock held by another process on {lock_path}: {err}"
                ) from err
        else:
            import fcntl

            try:
                fcntl.flock(file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, BlockingIOError) as err:
                raise SyncLockedError(
                    f"Lock held by another process on {lock_path}: {err}"
                ) from err

        # Record diagnostic info
        lock_file.seek(0)
        lock_file.truncate(0)
        diag_info = f"pid={os.getpid()}\ntimestamp={datetime.now(timezone.utc).isoformat()}\n"
        lock_file.write(diag_info.encode("utf-8"))
        lock_file.flush()

        logger.info("Acquired exclusive synchronization lock (%s, PID: %d).", lock_path, os.getpid())
        yield True

    finally:
        if lock_file and file_descriptor is not None:
            try:
                if os.name == "nt":
                    import msvcrt

                    lock_file.seek(0)
                    msvcrt.locking(file_descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(file_descriptor, fcntl.LOCK_UN)
            except Exception as unlock_err:
                logger.debug("Error releasing file lock: %s", unlock_err)

            try:
                lock_file.close()
            except Exception:
                pass

            try:
                if lock_path.exists():
                    lock_path.unlink(missing_ok=True)
            except Exception:
                pass

            logger.info("Released synchronization lock (%s).", lock_path)


def format_sync_summary(report: Dict[str, Any], failed_count: int = 0) -> str:
    """Format synchronization results into the standardized summary block."""
    is_dry_run = report.get("dry_run", False)
    sources_checked = report.get("total_sources_evaluated", 0)
    sources_unchanged = report.get("sources_unchanged", 0)
    sources_changed = report.get("sources_changed", 0)

    # Reprocessed sources are those that were either updated or planned for update
    reprocessed = sum(
        1
        for r in report.get("results", [])
        if r.get("action") in ("SYNCHRONIZED", "DRY_RUN_REPROCESS")
        or r.get("status") == "CHANGED"
    )

    embeddings_gen = report.get("total_embeddings_generated", 0)
    qdrant_deleted = report.get("total_points_deleted", 0)
    qdrant_upserted = report.get("total_points_upserted", 0)

    status_str = "SUCCESS" if failed_count == 0 else f"FAILED ({failed_count} errors)"
    if is_dry_run and failed_count == 0:
        status_str = "SUCCESS (DRY-RUN)"

    lines = [
        "",
        "==================================================",
        "BIS Sahayak Source Synchronization",
        "----------------------------------",
        f"Mode: {'DRY-RUN (no changes made)' if is_dry_run else 'LIVE'}",
        f"Sources checked: {sources_checked}",
        f"Unchanged: {sources_unchanged}",
        f"Changed: {sources_changed}",
        f"Reprocessed: {reprocessed}",
        f"Failed: {failed_count}",
        "",
        f"Embeddings generated: {embeddings_gen}",
        f"Qdrant deleted: {qdrant_deleted}",
        f"Qdrant upserted: {qdrant_upserted}",
        "",
        f"STATUS: {status_str}",
        "==================================================",
        "",
    ]
    return "\n".join(lines)


def run_sync(
    dry_run: bool = False,
    manifest_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    lock_file_path: Optional[str] = None,
    synchronizer: Optional[SourceSynchronizer] = None,
) -> int:
    """Execute the synchronization pipeline with lock protection and summary reporting.

    Args:
        dry_run: If True, inspects and plans without modifying disk or Qdrant.
        manifest_path: Optional path to custom sources.json.
        collection_name: Target collection name (defaults to 'bis_consumer').
        lock_file_path: Optional path to lock file.
        synchronizer: Optional pre-configured SourceSynchronizer instance.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    target_lock = (
        Path(lock_file_path)
        if lock_file_path
        else BACKEND_DIR / "data" / ".sync.lock"
    )

    try:
        with acquire_sync_lock(target_lock):
            logger.info(
                "Starting BIS source synchronization (collection: %s, dry_run: %s)...",
                collection_name or DEFAULT_COLLECTION_NAME,
                dry_run,
            )

            # Initialize synchronizer if not injected
            if synchronizer is None:
                synchronizer = SourceSynchronizer(
                    sources_manifest_path=manifest_path,
                    collection_name=collection_name,
                )

            # Run batch synchronization
            report = synchronizer.synchronize_all(
                collection_name=collection_name,
                dry_run=dry_run,
            )

            # Print human-readable summary
            summary = format_sync_summary(report, failed_count=0)
            print(summary)
            logger.info("Synchronization completed successfully.")
            return 0

    except SyncLockedError as lock_err:
        logger.warning(
            "Synchronization skipped: Another synchronization job is already running. %s",
            lock_err,
        )
        print("\n[NOTICE] Synchronization skipped: Another sync process is currently running.\n")
        return 0

    except Exception as exc:
        logger.error("Synchronization failed with error: %s", exc, exc_info=True)
        print(f"\n[ERROR] Synchronization failed: {exc}\n")
        return 1


def main() -> None:
    """Parse command-line arguments and run synchronization."""
    parser = argparse.ArgumentParser(
        description="BIS Sahayak Source Synchronization Runner (Step 4 Scheduler Entry Point)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate synchronization without modifying raw files, metadata, or Qdrant points.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to sources.json manifest file.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help="Target Qdrant collection name.",
    )
    parser.add_argument(
        "--lock-file",
        type=str,
        default=None,
        help="Path to custom lock file for concurrency control.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed DEBUG logging.",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    exit_code = run_sync(
        dry_run=args.dry_run,
        manifest_path=args.manifest,
        collection_name=args.collection,
        lock_file_path=args.lock_file,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
