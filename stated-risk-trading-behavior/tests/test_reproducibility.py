from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_required_code_files_exist() -> None:
    required = [
        "analysis/run_analysis.py",
        "analysis/run_external_validations.py",
        "analysis/run_robustness.py",
        "scripts/01_build_dataset.py",
        "scripts/02_run_primary_analysis.py",
        "scripts/03_run_robustness.py",
        "scripts/04_run_external_validation.py",
        "scripts/05_generate_outputs.py",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    assert missing == []


def test_generated_outputs_are_not_committed() -> None:
    generated = [
        "paper",
        "figures_final",
        "results_final",
        "results_external",
        "data/derived",
        "data/external",
    ]
    present = [path for path in generated if (ROOT / path).exists()]
    assert present == []


def test_default_config_loads() -> None:
    config = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    assert config
