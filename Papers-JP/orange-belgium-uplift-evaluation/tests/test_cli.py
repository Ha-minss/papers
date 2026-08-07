from pathlib import Path

import pytest

from churn_uplift.cli import build_parser
from churn_uplift.paths import prepare_output_dir


def test_cli_accepts_arbitrary_input_path_for_validation():
    args = build_parser().parse_args(
        ["validate-data", "--input", "/tmp/custom-name.csv"]
    )
    assert args.command == "validate-data"
    assert args.input == Path("/tmp/custom-name.csv")


def test_cli_requires_explicit_paths_for_run():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])


def test_prepare_output_dir_rejects_nonempty_directory_without_overwrite(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        prepare_output_dir(output, overwrite=False)


def test_prepare_output_dir_clears_nonempty_directory_with_overwrite(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "existing.txt").write_text("remove", encoding="utf-8")
    prepared = prepare_output_dir(output, overwrite=True)
    assert prepared == output
    assert list(output.iterdir()) == []
