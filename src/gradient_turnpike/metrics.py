"""Seed-paired metrics and median/IQR summaries for Table I."""

from __future__ import annotations

import math
from statistics import median
from typing import Iterable, MutableMapping, Sequence


TrainingComparisonRow = MutableMapping[str, object]


def quantile(values: Sequence[float], fraction: float) -> float:
    """Linearly interpolated empirical quantile for a nonempty sequence."""
    if not values or not 0.0 <= fraction <= 1.0:
        raise ValueError("values must be nonempty and fraction must lie in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return (ordered[lower] if lower == upper else
            ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def augment_paired_bptt_metrics(
    rows: Sequence[TrainingComparisonRow], *, expected_seeds: Iterable[int] | None = None,
) -> None:
    """Add BPTT loss gaps and timing ratios per seed, before aggregation."""
    references = {int(row["seed"]): row for row in rows if row["method"] == "BPTT"}
    if len(references) != sum(row["method"] == "BPTT" for row in rows):
        raise ValueError("each seed must have at most one BPTT reference row")
    if expected_seeds is not None and set(references) != set(expected_seeds):
        raise ValueError("a matching BPTT reference is required for every expected seed")
    for row in rows:
        seed = int(row["seed"])
        if seed not in references:
            raise ValueError(f"missing BPTT reference for seed {seed}")
        reference = references[seed]
        if row["method"] == "BPTT":
            row.update(bptt_reference_loss_gap=0.0, relative_bptt_reference_loss_gap=0.0,
                       runtime_relative_to_bptt=1.0, speedup_vs_bptt=1.0)
            continue
        full_training_mse = float(row["full_training_mse"])
        reference_mse = float(reference["full_training_mse"])
        runtime = float(row["training_time_seconds"])
        reference_runtime = float(reference["training_time_seconds"])
        if min(full_training_mse, reference_mse, runtime, reference_runtime) <= 0.0:
            raise ValueError("training MSE and runtime must be positive for paired metrics")
        loss_gap = full_training_mse - reference_mse
        row.update(bptt_reference_loss_gap=loss_gap,
                   relative_bptt_reference_loss_gap=loss_gap / reference_mse,
                   runtime_relative_to_bptt=runtime / reference_runtime,
                   speedup_vs_bptt=reference_runtime / runtime)


def summarize_training_comparison(rows: Sequence[TrainingComparisonRow]) -> tuple[list[dict], dict]:
    """Return median/IQR summaries from already seed-paired per-run rows."""
    groups: dict[tuple[str, str], list[TrainingComparisonRow]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["T_P"])), []).append(row)
    summary = []
    metrics = ("full_training_mse", "full_training_rmse", "bptt_reference_loss_gap", "relative_bptt_reference_loss_gap",
               "test_rmse", "training_time_seconds", "runtime_relative_to_bptt", "speedup_vs_bptt")
    for (method, context), group in groups.items():
        result: dict[str, object] = {"method": method, "T_P=T_F": context, "n_seeds": len(group)}
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            result[f"median_{metric}"] = median(values)
            result[f"iqr_{metric}"] = quantile(values, .75) - quantile(values, .25)
        summary.append(result)
    summary.sort(key=lambda row: (row["method"] != "BPTT", row["T_P=T_F"]))
    return summary, {"aggregation": "median and IQR across five seed-paired runs", "rows": summary}


def paired_ratio_median(
    values_by_seed: dict[int, float], references_by_seed: dict[int, float],
) -> float:
    """Median of seed-wise ratios, distinct from a ratio of medians."""
    if set(values_by_seed) != set(references_by_seed) or not values_by_seed:
        raise ValueError("values and references must contain the same nonempty seed set")
    ratios = []
    for seed, value in values_by_seed.items():
        reference = references_by_seed[seed]
        if not math.isfinite(value) or not math.isfinite(reference) or reference == 0.0:
            raise ValueError("paired ratios require finite values and nonzero references")
        ratios.append(value / reference)
    return median(ratios)
