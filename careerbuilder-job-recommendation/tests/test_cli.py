from pathlib import Path

import yaml

from careerrec.cli import build_parser, main


def _config(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    for name in ["apps.tsv", "users.tsv", "history.tsv", "test.tsv", "windows.tsv"]:
        (source / name).write_text("header\n", encoding="utf-8")
    (source / "jobs-1.zip").write_bytes(b"placeholder")
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "applications": "source/apps.tsv",
                    "users": "source/users.tsv",
                    "user_history": "source/history.tsv",
                    "test_users": "source/test.tsv",
                    "windows": "source/windows.tsv",
                    "job_archives": "source/jobs-*.zip",
                },
                "paths": {"workspace": "workspace"},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_parser_exposes_full_and_fast_reproduction_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for command in ["stage", "prepare", "semantic", "evaluate", "finalize", "run", "figures", "verify"]:
        assert command in help_text


def test_run_dry_run_prints_stages_without_executing(tmp_path: Path, capsys) -> None:
    config = _config(tmp_path)

    code = main(["run", "--config", str(config), "--dry-run"])

    assert code == 0
    output = capsys.readouterr().out
    assert "stage" in output
    assert "prepare" in output
    assert "semantic" in output
    assert "evaluate" in output
    assert "finalize" in output
    assert not (tmp_path / "workspace").exists()
