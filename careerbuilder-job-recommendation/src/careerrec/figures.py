from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_REQUIRED_SCHEMAS = {
    "paper_model_summary.csv": {"model", "ndcg10"},
    "ranking_diagnostics.csv": {"window_id", "model", "candidate_recall1000"},
    "paper_concentration_quartiles.csv": {
        "representation",
        "count_group",
        "concentration_quartile",
        "delta_ndcg",
    },
}


def _load(results_dir: Path, filename: str) -> pd.DataFrame:
    path = results_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = _REQUIRED_SCHEMAS[filename] - set(frame.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
    return frame


def _save(fig: plt.Figure, output_dir: Path, stem: str, formats: Iterable[str]) -> list[Path]:
    outputs: list[Path] = []
    for fmt in formats:
        if fmt not in {"pdf", "png"}:
            raise ValueError(f"Unsupported figure format: {fmt}")
        path = output_dir / f"{stem}.{fmt}"
        if fmt == "png":
            options = {"dpi": 300, "metadata": {"Software": "careerrec-paper"}}
        else:
            options = {
                "metadata": {
                    "Creator": "careerrec-paper",
                    "Producer": "Matplotlib",
                    "CreationDate": None,
                    "ModDate": None,
                }
            }
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03, **options)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.2,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.28,
        }
    )


def _candidate_recall(results_dir: Path) -> plt.Figure:
    data = _load(results_dir, "ranking_diagnostics.csv")
    data = data[data["window_id"].isin([5, 6, 7])]
    specs = [
        ("gte_global", "Global GTE", "o", "--"),
        ("gte_state", "State GTE", "s", "-."),
        ("gte_state_city", "State-city GTE", "^", "-"),
        ("itemknn", "Item-KNN", "D", ":"),
    ]
    fig, ax = plt.subplots(figsize=(3.45, 2.38))
    lines = []
    for key, label, marker, line_style in specs:
        series = data[data["model"] == key].sort_values("window_id")
        if series.empty:
            raise ValueError(f"ranking_diagnostics.csv has no rows for model {key}")
        (line,) = ax.plot(
            series["window_id"],
            series["candidate_recall1000"],
            label=label,
            marker=marker,
            linestyle=line_style,
            linewidth=1.35,
            markersize=4.2,
        )
        lines.append((line, label, float(series.iloc[-1]["candidate_recall1000"])))
    for line, label, y_value in lines:
        ax.text(7.08, y_value, label, va="center", ha="left", fontsize=7.0, color=line.get_color())
    ax.set_xlim(4.85, 7.78)
    ax.set_ylim(0.08, 0.55)
    ax.set_xticks([5, 6, 7])
    ax.set_xlabel("Test window")
    ax.set_ylabel("Candidate Recall@1000")
    ax.grid(axis="both")
    fig.subplots_adjust(left=0.18, right=0.72, bottom=0.20, top=0.97)
    return fig


def _model_performance(results_dir: Path) -> plt.Figure:
    data = _load(results_dir, "paper_model_summary.csv").set_index("model")
    rows = [
        ("BM25", "bm25"),
        ("Global GTE", "gte_global"),
        ("Localized GTE", "gte_state_city"),
        ("Count gate", "count_itemknn_gate"),
        ("Item-KNN", "itemknn"),
        ("Static hybrid", "static_itemknn_hybrid"),
    ]
    missing = [key for _, key in rows if key not in data.index]
    if missing:
        raise ValueError(f"paper_model_summary.csv is missing models: {missing}")
    labels = [label for label, _ in rows]
    values = np.array([float(data.loc[key, "ndcg10"]) for _, key in rows])
    fig, ax = plt.subplots(figsize=(3.45, 2.42))
    positions = np.arange(len(rows))
    bars = ax.barh(positions, values, edgecolor="black", linewidth=0.45, height=0.68)
    ax.set_yticks(positions, labels)
    ax.set_xlim(0, max(0.122, values.max() * 1.12))
    ax.set_xlabel("NDCG@10")
    ax.grid(axis="x")
    for rect, value in zip(bars, values):
        ax.text(
            value + 0.0022,
            rect.get_y() + rect.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=7.2,
        )
    relative = (values[-1] / values[-2] - 1.0) * 100.0
    ax.text(
        values[-1],
        len(rows) - 0.42,
        f"+{relative:.1f}% vs. Item-KNN",
        ha="right",
        va="bottom",
        fontsize=6.8,
    )
    fig.subplots_adjust(left=0.31, right=0.96, bottom=0.20, top=0.96)
    return fig


def _concentration_quartiles(results_dir: Path) -> plt.Figure:
    data = _load(results_dir, "paper_concentration_quartiles.csv")
    data = data[data["representation"] == "gte"].copy()
    row_order = ["2", "3-4", "5-9", "10+"]
    column_order = ["Q1", "Q2", "Q3", "Q4"]
    pivot = data.pivot(
        index="count_group",
        columns="concentration_quartile",
        values="delta_ndcg",
    )
    try:
        array = pivot.loc[row_order, column_order].to_numpy(dtype=float)
    except KeyError as exc:
        raise ValueError("Concentration table lacks one or more paper groups") from exc
    fig, ax = plt.subplots(figsize=(3.45, 2.48))
    image = ax.imshow(array, cmap="Blues", vmin=0.04, vmax=0.12, aspect="auto")
    ax.set_xticks(np.arange(4), column_order)
    ax.set_yticks(np.arange(4), row_order)
    ax.set_xlabel("GTE concentration quartile")
    ax.set_ylabel("Prior applications")
    ax.tick_params(length=0)
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            value = array[row, column]
            ax.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7.4,
                color="white" if value > 0.082 else "black",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.055, pad=0.035)
    colorbar.set_label("ΔNDCG@10\n(Item-KNN - content)", fontsize=7.0)
    colorbar.ax.tick_params(labelsize=6.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.22, right=0.88, bottom=0.21, top=0.97)
    return fig


def generate_figures(
    results_dir: str | Path,
    output_dir: str | Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> list[Path]:
    results_path = Path(results_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    _configure_style()

    figures = {
        "candidate_recall": _candidate_recall(results_path),
        "model_performance": _model_performance(results_path),
        "concentration_quartiles": _concentration_quartiles(results_path),
    }
    outputs: list[Path] = []
    for stem, figure in figures.items():
        outputs.extend(_save(figure, output_path, stem, formats))
    return sorted(outputs)
