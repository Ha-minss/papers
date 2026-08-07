from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    bootstrap_iterations: int = 2000
    sensitivity_repeats: int = 20
    random_pool_sizes: tuple[int, ...] = (10, 20, 50, 100)
    artifact_dir: Path = Path("artifacts")
    output_dir: Path = Path("artifacts/reproduced")
    headline_tolerance: float = 0.01
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        seed = int(payload.get("seed", payload.get("post_analysis_seed", 20260802)))
        cfg = cls(
            seed=seed,
            bootstrap_iterations=int(payload.get("bootstrap_iterations", 2000)),
            sensitivity_repeats=int(payload.get("sensitivity_repeats", 20)),
            random_pool_sizes=tuple(int(x) for x in payload.get("random_pool_sizes", [10, 20, 50, 100])),
            artifact_dir=Path(payload.get("artifact_dir", payload.get("paths", {}).get("artifact_dir", "artifacts"))),
            output_dir=Path(payload.get("output_dir", "artifacts/reproduced")),
            headline_tolerance=float(payload.get("headline_tolerance", 0.01)),
            raw=payload,
        )
        if cfg.bootstrap_iterations <= 0:
            raise ValueError("bootstrap_iterations must be positive")
        if any(size < 2 for size in cfg.random_pool_sizes):
            raise ValueError("random_pool_sizes must be >= 2")
        return cfg
