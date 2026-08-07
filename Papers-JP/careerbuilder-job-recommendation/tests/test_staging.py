from pathlib import Path
import zipfile

import yaml

from careerrec.config import load_config
from careerrec.staging import stage_dataset


def _write_tsv(path: Path, text: str = "header\n") -> None:
    path.write_text(text, encoding="utf-8")


def _write_zip(path: Path, member: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, "JobID\tWindowID\n")


def test_stage_dataset_accepts_arbitrary_source_names(tmp_path: Path) -> None:
    source = tmp_path / "source files"
    source.mkdir()
    files = {
        "applications": source / "interactions-export.tsv",
        "users": source / "people.tsv",
        "user_history": source / "career-history.tsv",
        "test_users": source / "evaluation-users.tsv",
        "windows": source / "time-windows.tsv",
    }
    for path in files.values():
        _write_tsv(path)
    _write_zip(source / "jobs-east.zip", "east.tsv")
    _write_zip(source / "jobs-west.zip", "west.tsv")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    **{key: str(path.relative_to(tmp_path)) for key, path in files.items()},
                    "job_archives": "source files/jobs-*.zip",
                },
                "paths": {"workspace": "portable-workspace"},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)

    staged = stage_dataset(config)

    assert staged == tmp_path / "portable-workspace" / "input"
    assert (staged / "apps.tsv").read_text(encoding="utf-8") == "header\n"
    assert (staged / "users.tsv").exists()
    assert (staged / "user_history.tsv").exists()
    assert (staged / "test_users.tsv").exists()
    assert (staged / "window_dates.tsv").exists()
    assert sorted(path.name for path in staged.glob("jobs_part*.zip")) == [
        "jobs_part1.zip",
        "jobs_part2.zip",
    ]
    assert (staged / "staging_manifest.json").exists()
