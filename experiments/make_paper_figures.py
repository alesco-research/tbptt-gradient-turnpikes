"""Render ECC Figures 2 and 3 from saved analysis tables, without recomputation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from gradient_turnpike.plotting import save_figure
from gradient_turnpike.utils import FIGURE_ROOT, OUTPUT_ROOT, load_config


DEFAULT_CONFIG = Path("configs/wh_paper.json")
mpl.rcParams["pdf.fonttype"] = 42


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing saved analysis data: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Saved analysis data are empty: {path}")
    return rows


def make_gradient_turnpike(config: dict, figure_root: Path) -> None:
    """Render Figure 2 from saved pointwise discrepancies."""
    source = OUTPUT_ROOT / "analysis" / "gradient_turnpike" / "pointwise" / "pointwise_gradient_errors.csv"
    rows = read_csv(source)
    contexts = config["gradient_turnpike"]["symmetric_contexts"]
    figure, axis = plt.subplots(figsize=(6.6, 3.25))
    for context in contexts:
        values = [row for row in rows if int(row["T_P"]) == context and int(row["T_F"]) == context]
        if not values:
            raise ValueError(f"Pointwise data lack T_P=T_F={context}; rerun analyze_gradient_turnpike.py")
        values.sort(key=lambda row: int(row["relative_t"]))
        axis.plot([int(row["relative_t"]) for row in values],
                  [float(row["gradient_error"]) for row in values], linewidth=1.35,
                  label=rf"$T_P=T_F={context}$")
    width = max(int(row["relative_t"]) for row in rows) + 1
    axis.axvline(0, color="0.35", linewidth=0.75, linestyle="--", zorder=0)
    axis.axvline(width - 1, color="0.35", linewidth=0.75, linestyle="--", zorder=0)
    axis.set(xlabel=r"$t$", ylabel=r"$\|g_t^{\mathrm{f}}-\hat g_t\|_2$", yscale="log",
             xlim=(-3, width + 3))
    axis.tick_params(labelsize=9)
    axis.legend(loc="upper center", ncol=3, fontsize=8.5, frameon=True,
                columnspacing=1.4, handlelength=2.2)
    save_figure(figure, figure_root / "gradient_turnpike")


def luminance(color: tuple[float, float, float, float]) -> float:
    red, green, blue, _ = color
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def format_discrepancy(value: float) -> str:
    """Use compact annotations while retaining original, non-log values."""
    if value >= 1:
        return f"{value:.2f}"
    if value >= 0.1:
        return f"{value:.2f}"
    if value >= 0.01:
        return f"{value:.3f}"
    return f"{value:.1e}"


def load_gradient_error_heatmap(config: dict) -> tuple[list[list[float]], list[int], list[int]]:
    """Validate the saved fixed-theta table and return its median matrix."""
    analysis_dir = OUTPUT_ROOT / "analysis" / "gradient_error_heatmap"
    aggregate_rows = read_csv(analysis_dir / "aggregated_gradient_errors.csv")
    protocol = config["fixed_parameter_gradient_analysis"]
    past_values = protocol["past_contexts"]
    future_values = protocol["future_contexts"]
    expected_cells = {(past, future) for past in past_values for future in future_values}
    aggregate = {(int(row["T_P"]), int(row["T_F"])): row for row in aggregate_rows}
    if set(aggregate) != expected_cells or len(aggregate_rows) != 25:
        raise ValueError("Gradient-error aggregate must contain exactly the 25 canonical context cells")
    expected_comparisons = len(config["reproducibility"]["seeds"]) * int(protocol["interval_count"])
    if any(int(row["sample_count"]) != expected_comparisons for row in aggregate_rows):
        raise ValueError("Each Figure-3 aggregate cell has an unexpected comparison count")
    matrix = [[float(aggregate[past, future]["median_absolute_error"])
               for past in past_values] for future in future_values]
    return matrix, past_values, future_values


def make_gradient_error_heatmap(config: dict, figure_root: Path) -> None:
    """Render Figure 3 from saved absolute-discrepancy aggregates."""
    values, past_values, future_values = load_gradient_error_heatmap(config)
    flat_values = [value for row in values for value in row]
    if min(flat_values) <= 0:
        raise ValueError("Figure-3 logarithmic color normalization requires positive discrepancies")
    norm = mpl.colors.LogNorm(vmin=min(flat_values), vmax=max(flat_values))
    cmap = plt.get_cmap("viridis")
    figure, axis = plt.subplots(figsize=(5.45, 4.3))
    image = axis.imshow(values, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    axis.set(xticks=range(len(past_values)), xticklabels=past_values,
             yticks=range(len(future_values)), yticklabels=future_values,
             xlabel=r"$T_P$", ylabel=r"$T_F$")
    axis.tick_params(labelsize=9)
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            color = "black" if luminance(cmap(norm(value))) > 0.52 else "white"
            axis.text(column_index, row_index, format_discrepancy(value), ha="center", va="center",
                      fontsize=8, color=color)
    figure.colorbar(image, ax=axis,
                    label=r"$\mathrm{median}\,\|G_{\mathcal{U}}^{\mathrm{f}}-\widehat G_{\mathcal{U}}\|_2$")
    save_figure(figure, figure_root / "gradient_error_heatmap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--figure", choices=("all", "gradient", "gradient-error"), default="all")
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_ROOT)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.figure in ("all", "gradient"):
        make_gradient_turnpike(config, args.figure_dir)
    if args.figure in ("all", "gradient-error"):
        make_gradient_error_heatmap(config, args.figure_dir)
    print(f"Saved requested paper figure(s) in {args.figure_dir}")


if __name__ == "__main__":
    main()
