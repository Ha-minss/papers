from __future__ import annotations

import runpy
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "build_dataset_only.py"
runpy.run_path(str(SCRIPT), run_name="__main__")
