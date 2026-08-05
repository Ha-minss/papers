import pytest
pytest.importorskip("torch")
from pathlib import Path
import zipfile

import pandas as pd
import yaml

from shore.config import ExperimentConfig
from shore.workflow import FullExperimentWorkflow


def make_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    output = root / "output"
    output.mkdir(parents=True)
    pd.DataFrame({"user_id": ["u1", "u2"], "resume_text": ["python", "sales"]}).to_csv(output / "users_clean.csv.gz", index=False)
    pd.DataFrame({"jd_no": ["j1"], "job_text": ["python engineer"], "job_split": ["test"]}).to_csv(output / "jobs_clean.csv.gz", index=False)
    pair = pd.DataFrame({"query_id": ["q", "q"], "jd_no": ["j1", "j1"], "user_id": ["u1", "u2"], "label": [1, 0], "candidate_source": ["applied_satisfied", "applied_rejected"]})
    pair.to_csv(output / "eval_confit_valid_100.csv.gz", index=False)
    pair.to_csv(output / "eval_confit_test_100.csv.gz", index=False)
    pair.to_csv(output / "eval_matched_pairwise_valid_10seeds.csv.gz", index=False)
    pair.to_csv(output / "eval_matched_pairwise_10seeds.csv.gz", index=False)
    pair.to_csv(output / "eval_stage_specific_valid_1plus2.csv.gz", index=False)
    pair.to_csv(output / "eval_stage_specific_1plus2.csv.gz", index=False)
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as z:
        for path in root.rglob("*"):
            if path.is_file():
                z.write(path, path.relative_to(root))
    return bundle


def test_workflow_prepare_and_bm25_stage(tmp_path):
    bundle = make_bundle(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "seed": 1,
        "candidate_count": 2,
        "paths": {"data_bundle": str(bundle), "artifact_dir": str(tmp_path / "artifacts")},
        "models": {"chinese_dense": "x", "qwen_embedding": "y", "gte_reranker": "z", "qwen_reranker": "q"},
    }))
    cfg = ExperimentConfig.from_yaml(config_path)
    workflow = FullExperimentWorkflow(cfg)
    prep = workflow.prepare_data({})
    assert Path(prep["output_dir"]).exists()
    result = workflow.run_bm25({})
    assert Path(result["score_path"]).exists()
    assert result["shape"] == [1, 2]
