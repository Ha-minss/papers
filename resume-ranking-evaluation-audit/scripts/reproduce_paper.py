from pathlib import Path
import argparse

from shore.config import ExperimentConfig
from shore.paper import reproduce_paper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_reproduction.yaml")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else cfg.artifact_dir
    output_dir = Path(args.output_dir) if args.output_dir else cfg.output_dir
    outputs = reproduce_paper(artifact_dir, output_dir)
    print(f"Created {len(outputs.files)} files under {outputs.output_dir}")


if __name__ == "__main__":
    main()
