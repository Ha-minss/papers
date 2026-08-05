from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .config import ProjectConfig, load_config
from .staging import stage_dataset


PIPELINE_STAGES = ("stage", "prepare", "semantic", "evaluate", "finalize")


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="YAML configuration file (default: configs/default.yaml)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="careerrec",
        description="Reproduce the job recommendation study and manuscript figures.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser("stage", help="Stage raw files into the canonical workspace")
    _add_config_argument(stage)
    stage.add_argument("--force", action="store_true", help="Replace staged copies")

    for name, help_text in [
        ("prepare", "Prepare text tables and embeddings"),
        ("semantic", "Build semantic candidates and concentration features"),
        ("evaluate", "Run behavioral recommenders and chronological evaluation"),
        ("finalize", "Run final gates, statistics, and paper tables"),
    ]:
        sub = subparsers.add_parser(name, help=help_text)
        _add_config_argument(sub)

    run = subparsers.add_parser("run", help="Execute the complete raw-data-to-results pipeline")
    _add_config_argument(run)
    run.add_argument("--force-stage", action="store_true", help="Replace staged raw files")
    run.add_argument("--dry-run", action="store_true", help="Print the stage plan only")

    figures = subparsers.add_parser("figures", help="Regenerate manuscript figures from aggregate CSV files")
    figures.add_argument("--results", type=Path, default=Path("results/paper"))
    figures.add_argument("--output", type=Path, default=Path("paper/figures"))
    figures.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png"),
        default=("pdf", "png"),
    )

    verify = subparsers.add_parser("verify", help="Check repository and reproduction assets")
    verify.add_argument("--root", type=Path, default=Path.cwd())

    return parser


def _prepare_config(config: ProjectConfig):
    from .pipeline.prepare_embeddings import RunConfig

    parameters = config.runtime.parameters
    return RunConfig(
        data_dir=str(config.paths.staged_data),
        output_dir=str(config.paths.artifacts),
        gte_model=config.models.content_encoder,
        jobbert_model=config.models.title_encoder,
        output_dtype=config.runtime.output_dtype,
        gte_max_length=int(parameters.get("content_max_length", 256)),
        jobbert_max_length=int(parameters.get("title_max_length", 64)),
        job_chunk_size=int(parameters.get("job_chunk_size", 50_000)),
        user_chunk_size=int(parameters.get("user_chunk_size", 25_000)),
        gte_batch_size=int(parameters.get("content_batch_size", 64)),
        jobbert_batch_size=int(parameters.get("title_batch_size", 128)),
    )


def _semantic_config(config: ProjectConfig):
    from .pipeline.semantic_candidates import Phase2Config

    parameters = config.runtime.parameters
    return Phase2Config(
        data_dir=str(config.paths.staged_data),
        output_dir=str(config.paths.artifacts),
        local_cache_dir=str(config.paths.cache / "semantic"),
        top_k=int(parameters.get("candidate_top_k", 1000)),
        query_batch_size=int(parameters.get("query_batch_size", 128)),
        concentration_chunk_size=int(parameters.get("concentration_chunk_size", 25_000)),
    )


def _evaluation_config(config: ProjectConfig):
    from .pipeline.recommendation_evaluation import Phase3Config

    parameters = config.runtime.parameters
    return Phase3Config(
        data_dir=str(config.paths.staged_data),
        output_dir=str(config.paths.artifacts),
        local_cache_dir=str(config.paths.cache / "evaluation"),
        recommendation_k=int(parameters.get("recommendation_k", 200)),
        evaluation_k=int(parameters.get("evaluation_k", 10)),
        bm25_recommendation_k=int(parameters.get("candidate_top_k", 1000)),
        bm25_search_k=int(parameters.get("bm25_search_k", 1500)),
        bm25_query_batch_size=int(parameters.get("bm25_query_batch_size", 256)),
        itemknn_k=int(parameters.get("itemknn_neighbors", 100)),
        itemknn_search_k=int(parameters.get("itemknn_search_k", 700)),
        als_factors=int(parameters.get("als_factors", 64)),
        als_regularization=float(parameters.get("als_regularization", 0.03)),
        als_alpha=float(parameters.get("als_alpha", 20.0)),
        als_iterations=int(parameters.get("als_iterations", 15)),
        behavior_batch_size=int(parameters.get("behavior_batch_size", 512)),
        bootstrap_repetitions=int(parameters.get("bootstrap_repetitions", 1000)),
        random_seed=config.runtime.random_seed,
    )


def _final_config(config: ProjectConfig):
    from .pipeline.final_analysis import Phase4Config

    parameters = config.runtime.parameters
    return Phase4Config(
        data_dir=str(config.paths.staged_data),
        output_dir=str(config.paths.artifacts),
        local_cache_dir=str(config.paths.cache / "final"),
        localized_top_k=int(parameters.get("candidate_top_k", 1000)),
        recommendation_k=int(parameters.get("recommendation_k", 200)),
        evaluation_k=int(parameters.get("evaluation_k", 10)),
        query_batch_size=int(parameters.get("query_batch_size", 128)),
        min_local_pool=int(parameters.get("min_local_pool", 100)),
        bootstrap_repetitions=int(parameters.get("bootstrap_repetitions", 1000)),
        random_seed=config.runtime.random_seed,
    )


def _run_stage(name: str, config: ProjectConfig) -> object:
    if name == "stage":
        return stage_dataset(config)
    if name == "prepare":
        from .pipeline.prepare_embeddings import run_all

        return run_all(_prepare_config(config))
    if name == "semantic":
        from .pipeline.semantic_candidates import run_phase2

        return run_phase2(_semantic_config(config))
    if name == "evaluate":
        from .pipeline.recommendation_evaluation import run_phase3

        return run_phase3(_evaluation_config(config))
    if name == "finalize":
        from .pipeline.final_analysis import run_phase4

        return run_phase4(_final_config(config))
    raise ValueError(f"Unknown pipeline stage: {name}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "figures":
            from .figures import generate_figures

            outputs = generate_figures(args.results, args.output, tuple(args.formats))
            for output in outputs:
                print(output)
            return 0
        if args.command == "verify":
            from .verification import verify_repository

            report = verify_repository(args.root)
            print(json.dumps(report, indent=2))
            return 0

        config = load_config(args.config)
        if args.command == "run":
            if args.dry_run:
                print("Pipeline plan: " + " -> ".join(PIPELINE_STAGES))
                return 0
            stage_dataset(config, force=args.force_stage)
            for stage_name in PIPELINE_STAGES[1:]:
                print(f"\n=== {stage_name} ===")
                _run_stage(stage_name, config)
            return 0
        if args.command == "stage":
            print(stage_dataset(config, force=args.force))
            return 0
        _run_stage(args.command, config)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
