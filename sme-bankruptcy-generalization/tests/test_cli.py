import importlib
import subprocess
import sys

CLI_MODULES = (
    "scripts.prepare_data",
    "scripts.run_factorial_experiments",
    "scripts.run_structure_experiments",
    "scripts.run_primary_predictions",
    "scripts.run_bootstrap",
    "scripts.run_partial_pooling",
    "scripts.run_partial_pool_bootstrap",
    "scripts.run_diagnostics",
)


def test_cli_modules_expose_argument_parsers():
    for module_name in CLI_MODULES:
        module = importlib.import_module(module_name)
        parser = module.build_parser()
        assert parser.prog


def test_all_cli_help_commands_exit_successfully():
    for module_name in CLI_MODULES:
        completed = subprocess.run(
            [sys.executable, "-m", module_name, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.lower()
