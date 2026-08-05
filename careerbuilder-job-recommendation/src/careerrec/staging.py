from __future__ import annotations

import glob
import hashlib
import json
import shutil
from pathlib import Path

from .config import ProjectConfig


_CANONICAL_FILES = {
    "applications": "apps.tsv",
    "users": "users.tsv",
    "user_history": "user_history.tsv",
    "test_users": "test_users.tsv",
    "windows": "window_dates.tsv",
}


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source: Path, target: Path, force: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        if target.stat().st_size == source.stat().st_size and _sha256(target) == _sha256(source):
            return
    shutil.copy2(source, target)


def stage_dataset(config: ProjectConfig, force: bool = False) -> Path:
    destination = config.paths.staged_data
    destination.mkdir(parents=True, exist_ok=True)

    source_files = {
        "applications": config.data.applications,
        "users": config.data.users,
        "user_history": config.data.user_history,
        "test_users": config.data.test_users,
        "windows": config.data.windows,
    }
    manifest_files: list[dict[str, object]] = []
    for key, source in source_files.items():
        target = destination / _CANONICAL_FILES[key]
        _copy(source, target, force=force)
        manifest_files.append(
            {
                "role": key,
                "source": str(source),
                "staged": target.name,
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )

    archives = [Path(path) for path in sorted(glob.glob(str(config.data.job_archives_pattern)))]
    if not archives:
        raise FileNotFoundError(
            f"No job archives matched: {config.data.job_archives_pattern}"
        )
    for stale in destination.glob("jobs_part*.zip"):
        if force or stale.name not in {f"jobs_part{i}.zip" for i in range(1, len(archives) + 1)}:
            stale.unlink()
    for index, source in enumerate(archives, start=1):
        target = destination / f"jobs_part{index}.zip"
        _copy(source, target, force=force)
        manifest_files.append(
            {
                "role": "job_archive",
                "source": str(source),
                "staged": target.name,
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )

    manifest = {
        "status": "complete",
        "config": str(config.config_path),
        "destination": str(destination),
        "files": manifest_files,
    }
    (destination / "staging_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination
