import hashlib

from churn_uplift.provenance import build_run_metadata, file_sha256


def test_file_sha256_matches_standard_library(tmp_path):
    path = tmp_path / "arbitrary.csv"
    path.write_bytes(b"y,t,PC1\n0,1,0.5\n")
    assert file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_run_metadata_records_resolved_paths_and_hashes(tmp_path):
    data = tmp_path / "data.csv"
    config = tmp_path / "paper.yaml"
    data.write_text("y,t,PC1\n0,1,0.5\n", encoding="utf-8")
    config.write_text("random_seed: 42\n", encoding="utf-8")
    metadata = build_run_metadata(data, config)
    assert metadata["data"]["path"] == str(data.resolve())
    assert metadata["config"]["path"] == str(config.resolve())
    assert len(metadata["data"]["sha256"]) == 64
    assert metadata["runtime"]["python"]
