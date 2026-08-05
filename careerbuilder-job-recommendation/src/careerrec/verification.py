from __future__ import annotations

from pathlib import Path


_REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "pyproject.toml",
    "Makefile",
    "requirements.txt",
    "configs/default.yaml",
    "src/careerrec/cli.py",
    "src/careerrec/config.py",
    "src/careerrec/staging.py",
    "src/careerrec/pipeline/final_analysis.py",
    "src/careerrec/pipeline/prepare_embeddings.py",
    "src/careerrec/pipeline/recommendation_evaluation.py",
    "src/careerrec/pipeline/semantic_candidates.py",
]

_UNWANTED_PATHS = [
    ".github",
    "docs",
    "paper",
    "results",
    "data",
    "notebooks",
]

_FORBIDDEN_SUFFIXES = {".csv", ".gz", ".pdf", ".png", ".ipynb", ".pyc"}
_FORBIDDEN_TOKENS = (
    "/content/",
    "MyDrive/",
    "careerbuilder_colab",
    "CareerBuilder_Phase",
)


def _text_files(root: Path) -> list[Path]:
    candidates = [root / "src", root / "configs", root / "tests"]
    files: list[Path] = []
    for directory in candidates:
        if directory.exists():
            files.extend(
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".yml", ".toml", ".md"}
            )
    files.extend(
        path
        for path in [root / "README.md", root / "Makefile", root / "pyproject.toml"]
        if path.exists()
    )
    return sorted(set(files))


def verify_repository(root: str | Path) -> dict[str, object]:
    root_path = Path(root).expanduser().resolve()
    missing = [relative for relative in _REQUIRED_FILES if not (root_path / relative).exists()]
    unwanted_paths = [relative for relative in _UNWANTED_PATHS if (root_path / relative).exists()]
    forbidden_files = [
        str(path.relative_to(root_path))
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in _FORBIDDEN_SUFFIXES
    ]

    forbidden_tokens: list[dict[str, str]] = []
    for path in _text_files(root_path):
        if path.name == "verification.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in _FORBIDDEN_TOKENS:
            if token in text:
                forbidden_tokens.append({"file": str(path.relative_to(root_path)), "token": token})

    status = "complete" if not missing and not unwanted_paths and not forbidden_files and not forbidden_tokens else "incomplete"
    return {
        "status": status,
        "root": str(root_path),
        "missing_files": missing,
        "unwanted_paths": unwanted_paths,
        "forbidden_files": forbidden_files,
        "forbidden_path_matches": forbidden_tokens,
    }
