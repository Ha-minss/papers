from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ROOT = {
    "README.md",
    "requirements.txt",
    "Makefile",
    "LICENSE",
    "CITATION.cff",
    "config",
    "scripts",
    "tests",
}
FORBIDDEN_DIR_NAMES = {
    "paper",
    "docs",
    "data",
    "analysis",
    "results",
    "results_final",
    "results_external",
    "figures",
    "figures_final",
    "notebooks",
}
FORBIDDEN_SUFFIXES = {".ipynb", ".csv", ".gz", ".pdf", ".png"}
FORBIDDEN_FILES = {"requirements-lock.txt", "SHA256SUMS.txt", ".gitignore"}


class RepositoryContractTests(unittest.TestCase):
    def test_root_contains_only_code_portfolio_surface(self):
        actual = {path.name for path in ROOT.iterdir() if path.name != "__pycache__"}
        self.assertEqual(actual, REQUIRED_ROOT)

    def test_forbidden_artifacts_are_absent(self):
        for path in ROOT.rglob("*"):
            if "__pycache__" in path.parts or path.name.endswith(".pyc"):
                continue
            self.assertNotIn(path.name, FORBIDDEN_FILES, str(path))
            if path.is_dir():
                self.assertNotIn(path.name, FORBIDDEN_DIR_NAMES, str(path))
            if path.is_file():
                self.assertNotIn(path.suffix.lower(), FORBIDDEN_SUFFIXES, str(path))

    def test_default_config_is_code_only_and_points_outside_repo(self):
        config = json.loads((ROOT / "config" / "default.json").read_text(encoding="utf-8"))
        self.assertTrue(config["paths"]["prepared_data_json"].startswith("../"))
        self.assertTrue(config["paths"]["results_root"].startswith("../"))
        self.assertEqual(config["stage1"]["seeds"], 30)
        self.assertEqual(config["stage2"]["alpha_grid"], [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(config["stage3"]["scenario_count"], 100)


if __name__ == "__main__":
    unittest.main()
