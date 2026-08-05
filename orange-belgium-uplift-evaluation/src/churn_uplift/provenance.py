from __future__ import annotations

import hashlib
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "catboost",
    "optuna",
    "matplotlib",
    "pyyaml",
)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def build_run_metadata(data_path: str | Path, config_path: str | Path) -> dict[str, Any]:
    data = Path(data_path).resolve()
    config = Path(config_path).resolve()
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "path": str(data),
            "sha256": file_sha256(data),
            "bytes": data.stat().st_size,
        },
        "config": {
            "path": str(config),
            "sha256": file_sha256(config),
            "bytes": config.stat().st_size,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
    }
