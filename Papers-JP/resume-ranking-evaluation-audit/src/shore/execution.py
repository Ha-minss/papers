from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


class ExperimentRunner:
    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.manifest_dir = self.artifact_dir / "manifests"
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self._stages: dict[str, Callable[[dict], dict]] = {}

    def register(self, name: str, function: Callable[[dict], dict]) -> None:
        if name in self._stages:
            raise ValueError(f"stage already registered: {name}")
        self._stages[name] = function

    def run_stage(self, name: str, context: dict, force: bool = False) -> dict:
        if name not in self._stages:
            raise KeyError(f"unknown stage: {name}")
        manifest = self.manifest_dir / f"{name}.json"
        if manifest.exists() and not force:
            return json.loads(manifest.read_text(encoding="utf-8"))
        result = self._stages[name](context)
        payload = {"stage": name, "status": "complete", **result}
        temp = manifest.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        temp.replace(manifest)
        return payload

    def run(self, stages: list[str], context: dict, force: bool = False) -> dict[str, dict]:
        return {stage: self.run_stage(stage, context, force=force) for stage in stages}
