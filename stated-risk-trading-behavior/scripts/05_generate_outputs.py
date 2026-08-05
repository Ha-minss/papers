from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name in ("save_action_predictions.py", "save_m4_importance.py", "finalize_outputs.py"):
    runpy.run_path(str(ROOT / "analysis" / name), run_name="__main__")
