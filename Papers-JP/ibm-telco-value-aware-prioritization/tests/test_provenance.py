import hashlib

from telco_churn.provenance import build_run_metadata, file_sha256


def test_file_sha256_matches_standard_library(tmp_path):
    path = tmp_path / "arbitrary.csv"
    path.write_bytes(b"CustomerID,Churn Value\nA,0\n")
    assert file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_run_metadata_records_resolved_paths_and_hashes(tmp_path):
    data = tmp_path / "data.csv"
    config = tmp_path / "paper.yaml"
    data.write_text("CustomerID,Churn Value\nA,0\n", encoding="utf-8")
    config.write_text("random_seed: 42\n", encoding="utf-8")
    metadata = build_run_metadata(data, config)
    assert metadata["data"]["path"] == str(data.resolve())
    assert metadata["config"]["path"] == str(config.resolve())
    assert len(metadata["config"]["sha256"]) == 64
    assert metadata["runtime"]["platform"]
