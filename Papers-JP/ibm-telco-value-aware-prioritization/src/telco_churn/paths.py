from __future__ import annotations

import shutil
from pathlib import Path


def prepare_output_dir(path: str | Path, *, overwrite: bool = False) -> Path:
    """Create an output directory without silently replacing prior results."""
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output}. "
                "Choose another directory or pass --overwrite."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output
