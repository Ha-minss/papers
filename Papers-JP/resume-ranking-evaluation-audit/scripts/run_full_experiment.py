"""Run the complete GPU experiment with restart-safe stage manifests."""
from __future__ import annotations

import argparse

from shore.config import ExperimentConfig
from shore.execution import ExperimentRunner
from shore.workflow import FullExperimentWorkflow


DEFAULT_STAGES = [
    "prepare_data",
    "run_bm25",
    "run_chinese_dense",
    "run_qwen_embedding",
    "run_hybrid_rrf",
    "run_gte_reranker",
    "run_qwen_reranker",
    "evaluate_zero_shot",
    "run_pointwise_adaptation",
    "run_pairwise_adaptation",
    "run_preservation_adaptation",
    "run_paper_analysis",
]


def build_runner(cfg: ExperimentConfig) -> ExperimentRunner:
    workflow = FullExperimentWorkflow(cfg)
    runner = ExperimentRunner(cfg.artifact_dir)
    for name in DEFAULT_STAGES:
        runner.register(name, getattr(workflow, name))
    return runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/full_experiment.yaml")
    parser.add_argument("--force", action="store_true", help="rerun stages even when manifests exist")
    parser.add_argument("--stages", nargs="+", choices=DEFAULT_STAGES, help="run only selected stages in the given order")
    parser.add_argument("--skip-adaptation", action="store_true", help="run zero-shot systems and paper analysis only")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    stages = args.stages or list(DEFAULT_STAGES)
    if args.skip_adaptation:
        stages = [s for s in stages if "adaptation" not in s]
    results = build_runner(cfg).run(stages, context={}, force=args.force)
    for name, result in results.items():
        print(f"{name}: {result['status']}")


if __name__ == "__main__":
    main()
