from careerrec.pipeline.prepare_embeddings import RunConfig
from careerrec.pipeline.semantic_candidates import Phase2Config
from careerrec.pipeline.recommendation_evaluation import Phase3Config
from careerrec.pipeline.final_analysis import Phase4Config


def test_pipeline_modules_have_portable_cache_defaults() -> None:
    configs = [
        RunConfig(data_dir="data", output_dir="artifacts"),
        Phase2Config(data_dir="data", output_dir="artifacts"),
        Phase3Config(data_dir="data", output_dir="artifacts"),
        Phase4Config(data_dir="data", output_dir="artifacts"),
    ]
    for config in configs[1:]:
        assert not str(config.local_cache_dir).startswith("/content")
        assert not str(config.local_cache_dir).startswith("/home/")
