from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


class Pipeline:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = Path(artifact_dir)
        self.manifest_dir = self.artifact_dir / "manifests"
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def run_stage(self, name: str, runner: Callable[[], dict], force: bool = False) -> dict:
        manifest = self.manifest_dir / f"{name}.json"
        if manifest.exists() and not force:
            return json.loads(manifest.read_text(encoding="utf-8"))
        result = runner()
        temp = manifest.with_suffix(".json.tmp")
        temp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        temp.replace(manifest)
        return result
