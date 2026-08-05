"""Lightweight integrity checks for the code repository."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    ROOT / "README.md",
    ROOT / "requirements.txt",
    ROOT / "config" / "default.yaml",
    ROOT / "DATA_LICENSE.md",
    ROOT / "analysis" / "run_analysis.py",
    ROOT / "analysis" / "run_external_validations.py",
    ROOT / "analysis" / "run_robustness.py",
    ROOT / "scripts" / "01_build_dataset.py",
    ROOT / "scripts" / "02_run_primary_analysis.py",
    ROOT / "scripts" / "03_run_robustness.py",
    ROOT / "scripts" / "04_run_external_validation.py",
    ROOT / "scripts" / "05_generate_outputs.py",
]

GENERATED_PATHS = [
    ROOT / "paper",
    ROOT / "figures_final",
    ROOT / "results_final",
    ROOT / "results_external",
    ROOT / "data" / "derived",
    ROOT / "data" / "external",
]


def main() -> None:
    missing = [path for path in REQUIRED_PATHS if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path.relative_to(ROOT)}" for path in missing)
        raise FileNotFoundError(f"Required repository files are missing:\n{formatted}")

    present_generated = [path for path in GENERATED_PATHS if path.exists()]
    if present_generated:
        formatted = "\n".join(f"- {path.relative_to(ROOT)}" for path in present_generated)
        raise AssertionError(f"Generated output directories should not be committed:\n{formatted}")

    print("Repository structure check passed.")


if __name__ == "__main__":
    main()
