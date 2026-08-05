from __future__ import annotations

import runpy
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "run_robustness.py"
runpy.run_path(str(SCRIPT), run_name="__main__")
