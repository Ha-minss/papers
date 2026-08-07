"""Lightweight integrity checks for the code repository."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    ROOT / "README.md",
    ROOT / "requirements.txt",
    ROOT / "Makefile",
    ROOT / "LICENSE",
    ROOT / "CITATION.cff",
    ROOT / "config" / "default.yaml",
    ROOT / "scripts" / "01_build_dataset.py",
    ROOT / "scripts" / "02_run_primary_analysis.py",
    ROOT / "scripts" / "03_run_robustness.py",
    ROOT / "scripts" / "04_run_external_validation.py",
    ROOT / "scripts" / "05_generate_outputs.py",
]

REMOVED_PATHS = [
    ROOT / "analysis",
    ROOT / "docs",
    ROOT / "paper",
    ROOT / "figures_final",
    ROOT / "results_final",
    ROOT / "results_external",
    ROOT / "data",
]


def main() -> None:
    missing = [path for path in REQUIRED_PATHS if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path.relative_to(ROOT)}" for path in missing)
        raise FileNotFoundError(f"Required repository files are missing:\n{formatted}")

    present_removed = [path for path in REMOVED_PATHS if path.exists()]
    if present_removed:
        formatted = "\n".join(f"- {path.relative_to(ROOT)}" for path in present_removed)
        raise AssertionError(f"Unexpected extra directories are present:\n{formatted}")

    print("Repository structure check passed.")


if __name__ == "__main__":
    main()
