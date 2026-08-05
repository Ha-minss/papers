from pathlib import Path

import yaml

from careerrec.config import load_config


def test_load_config_resolves_relative_paths_from_config_directory(tmp_path: Path) -> None:
    config_dir = tmp_path / "project"
    config_dir.mkdir()
    raw_dir = config_dir / "inputs"
    raw_dir.mkdir()
    config_path = config_dir / "local.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "applications": "inputs/custom-applications.tsv",
                    "users": "inputs/custom-users.tsv",
                    "user_history": "inputs/history.tsv",
                    "test_users": "inputs/test-users.tsv",
                    "windows": "inputs/windows.tsv",
                    "job_archives": "inputs/job-*.zip",
                },
                "paths": {
                    "workspace": "workspace",
                    "artifacts": "workspace/artifacts",
                    "cache": "workspace/cache",
                    "paper_results": "results/paper",
                    "paper_figures": "paper/figures",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.data.applications == raw_dir / "custom-applications.tsv"
    assert config.data.job_archives_pattern == raw_dir / "job-*.zip"
    assert config.paths.workspace == config_dir / "workspace"
    assert config.paths.artifacts == config_dir / "workspace" / "artifacts"
    assert config.paths.paper_results == config_dir / "results" / "paper"
