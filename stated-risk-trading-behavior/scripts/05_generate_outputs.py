from __future__ import annotations

import runpy
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
for name in ("save_action_predictions.py", "save_m4_importance.py", "finalize_outputs.py"):
    runpy.run_path(str(SCRIPTS / name), run_name="__main__")
