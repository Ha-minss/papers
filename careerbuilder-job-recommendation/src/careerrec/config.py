from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


@dataclass(frozen=True)
class DataConfig:
    applications: Path
    users: Path
    user_history: Path
    test_users: Path
    windows: Path
    job_archives_pattern: Path


@dataclass(frozen=True)
class PathsConfig:
    workspace: Path
    artifacts: Path
    cache: Path
    paper_results: Path
    paper_figures: Path

    @property
    def staged_data(self) -> Path:
        return self.workspace / "input"


@dataclass(frozen=True)
class ModelsConfig:
    content_encoder: str = "Alibaba-NLP/gte-modernbert-base"
    title_encoder: str = "TechWolf/JobBERT-v2"


@dataclass(frozen=True)
class RuntimeConfig:
    random_seed: int = 42
    device: str = "auto"
    output_dtype: str = "float16"
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectConfig:
    config_path: Path
    data: DataConfig
    paths: PathsConfig
    models: ModelsConfig
    runtime: RuntimeConfig


def _require(mapping: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in mapping or mapping[key] in (None, ""):
        raise ValueError(f"Missing required configuration value: {section}.{key}")
    return mapping[key]


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("The configuration root must be a mapping")

    base = config_path.parent
    data_raw = raw.get("data", {}) or {}
    paths_raw = raw.get("paths", {}) or {}
    models_raw = raw.get("models", {}) or {}
    runtime_raw = raw.get("runtime", {}) or {}

    data = DataConfig(
        applications=_resolve(base, _require(data_raw, "applications", "data")),
        users=_resolve(base, _require(data_raw, "users", "data")),
        user_history=_resolve(base, _require(data_raw, "user_history", "data")),
        test_users=_resolve(base, _require(data_raw, "test_users", "data")),
        windows=_resolve(base, _require(data_raw, "windows", "data")),
        job_archives_pattern=_resolve(base, _require(data_raw, "job_archives", "data")),
    )

    workspace = _resolve(base, paths_raw.get("workspace", "workspace"))
    paths = PathsConfig(
        workspace=workspace,
        artifacts=_resolve(base, paths_raw.get("artifacts", workspace / "artifacts")),
        cache=_resolve(base, paths_raw.get("cache", workspace / "cache")),
        paper_results=_resolve(base, paths_raw.get("paper_results", "results/paper")),
        paper_figures=_resolve(base, paths_raw.get("paper_figures", "paper/figures")),
    )

    models = ModelsConfig(
        content_encoder=str(models_raw.get("content_encoder", ModelsConfig.content_encoder)),
        title_encoder=str(models_raw.get("title_encoder", ModelsConfig.title_encoder)),
    )
    runtime = RuntimeConfig(
        random_seed=int(runtime_raw.get("random_seed", 42)),
        device=str(runtime_raw.get("device", "auto")),
        output_dtype=str(runtime_raw.get("output_dtype", "float16")),
        parameters=dict(runtime_raw.get("parameters", {}) or {}),
    )
    return ProjectConfig(
        config_path=config_path,
        data=data,
        paths=paths,
        models=models,
        runtime=runtime,
    )
