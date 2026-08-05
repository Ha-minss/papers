# When Should Application History Be Trusted in Job Recommendation?

Code repository for **When Should Application History Be Trusted in Job Recommendation?**

This repository keeps the implementation for a chronological job-recommendation study comparing profile/content signals with observed application behavior. Generated result tables, figures, manuscript files, notebooks, and raw data are intentionally not committed.

## Repository Map

```text
configs/        Portable YAML configuration
src/careerrec/  Importable pipeline package and CLI
tests/          Lightweight unit and repository checks
```

## Data

Raw CareerBuilder Job Recommendation Challenge data are not redistributed. Put the raw files anywhere on your machine, copy `configs/default.yaml` to a local config file, and update the paths under `data:`.

```bash
cp configs/default.yaml configs/local.yaml
python -m careerrec run --config configs/local.yaml
```

Generated workspaces, result tables, figures, and paper outputs should be written locally and kept out of Git.

## Environment

Python 3.10+ is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-build-isolation -e ".[dev]"
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --no-build-isolation -e ".[dev]"
```

## Commands

```bash
python -m careerrec stage --config configs/local.yaml
python -m careerrec prepare --config configs/local.yaml
python -m careerrec semantic --config configs/local.yaml
python -m careerrec evaluate --config configs/local.yaml
python -m careerrec finalize --config configs/local.yaml
python -m careerrec run --config configs/local.yaml
python -m careerrec verify --root .
```

With Make installed:

```bash
make test
make verify
make reproduce CONFIG=configs/local.yaml
```

## Check

```bash
pytest
python -m careerrec verify --root .
```

These checks validate the code-only repository structure and configuration. They do not require raw data or generated outputs.

## License

Code is released under the MIT License. Citation metadata are provided in `CITATION.cff`.
