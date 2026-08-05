from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_required_top_level_items() -> None:
    required = [
        "README.md",
        "requirements.txt",
        "Makefile",
        "LICENSE",
        "CITATION.cff",
        "config",
        "scripts",
        "tests",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    assert missing == []


def test_unwanted_directories_are_absent() -> None:
    unwanted = [
        "analysis",
        "docs",
        "data",
        "paper",
        "figures_final",
        "results_final",
        "results_external",
    ]
    present = [path for path in unwanted if (ROOT / path).exists()]
    assert present == []


def test_default_config_loads() -> None:
    config = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    assert config
