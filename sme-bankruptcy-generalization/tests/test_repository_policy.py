from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TOP_LEVEL = {
    "README.md",
    "requirements.txt",
    "Makefile",
    "LICENSE",
    "CITATION.cff",
    "config",
    "scripts",
    "tests",
}
FORBIDDEN_DIRECTORIES = {
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


def test_top_level_is_code_only():
    entries = {path.name for path in ROOT.iterdir() if path.name != ".git"}
    assert entries == REQUIRED_TOP_LEVEL


def test_forbidden_directories_and_artifacts_are_absent():
    for path in ROOT.rglob("*"):
        assert not any(part in FORBIDDEN_DIRECTORIES for part in path.relative_to(ROOT).parts)
        assert path.name not in FORBIDDEN_FILES
        if path.is_file():
            assert path.suffix.lower() not in FORBIDDEN_SUFFIXES


def test_documented_workflow_keeps_local_environment_and_bytecode_outside_repo():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "python -m venv .venv" not in readme
    assert "python -m venv ../.venv-sme-bankruptcy" in readme
    assert "python -B -m scripts." in readme
    assert "RUN := PYTHONDONTWRITEBYTECODE=1 $(PYTHON)" in makefile
