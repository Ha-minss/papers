from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ScriptSmokeTests(unittest.TestCase):
    def test_core_modules_import_without_data_files(self):
        from scripts.pom_v4 import controlled_lp_v4  # noqa: F401
        from scripts.pom_v4 import predictive_scoring  # noqa: F401
        from scripts.pom_v4 import run_alpha_sweep  # noqa: F401
        from scripts.pom_v4 import run_structural_ablation  # noqa: F401
        from scripts.pom_v4 import structural_architecture  # noqa: F401

    def test_pure_scoring_primitives_work_on_synthetic_arrays(self):
        from scripts.pom_v4.predictive_scoring import energy_score, marginal_crps, variogram_score

        observation = np.array([2.0, 5.0, 7.0])
        ensemble = np.repeat(observation[None, :], 4, axis=0)
        scale = np.ones(3)
        self.assertEqual(energy_score(ensemble, observation, scale), 0.0)
        self.assertEqual(variogram_score(ensemble, observation, scale), 0.0)
        self.assertEqual(marginal_crps(ensemble, observation, scale), 0.0)

    def test_architecture_contract_is_data_free(self):
        from scripts.pom_v4.structural_architecture import ArchitectureSpec

        self.assertEqual(ArchitectureSpec.pooling(0.25).alpha, 0.25)
        with self.assertRaises(ValueError):
            ArchitectureSpec.pooling(1.01)

    def test_runner_cli_help_does_not_require_data(self):
        modules = [
            "scripts.pom_v4.run_structural_ablation",
            "scripts.pom_v4.run_alpha_sweep",
            "scripts.pom_v4.predictive_scoring",
        ]
        for module in modules:
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
