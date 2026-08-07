from pathlib import Path

from careerrec.verification import verify_repository


def test_repository_is_code_only() -> None:
    root = Path(__file__).resolve().parents[1]

    report = verify_repository(root)

    assert report["status"] == "complete"
    assert report["missing_files"] == []
    assert report["unwanted_paths"] == []
    assert report["forbidden_files"] == []
    assert report["forbidden_path_matches"] == []
