"""Compute Figure 3 median update-gradient errors at five fixed BPTT models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median

import torch

from gradient_turnpike.data import load_wh
from gradient_turnpike.evaluation import load_model
from gradient_turnpike.gradients import flatten_parameter_gradient, full_context_loss, sample_update_starts, tbptt_loss
from gradient_turnpike.training import prepare_sequences
from gradient_turnpike.utils import OUTPUT_ROOT, load_config, prepare_output_directory, resolve_device


DEFAULT_CONFIG = Path("configs/wh_paper.json")
ANALYSIS_DIRECTORY = OUTPUT_ROOT / "analysis" / "gradient_error_heatmap"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a nonempty compact analysis table."""
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require_final_bptt_checkpoint(metadata: dict, checkpoint: Path, *, seed: int, max_updates: int) -> None:
    """Reject checkpoints that do not belong to the fixed-parameter protocol."""
    expected = {"training_method": "bptt", "seed": seed, "checkpoint_update": max_updates}
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"{checkpoint}: expected {field}={value!r}, found {metadata.get(field)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    protocol = config["fixed_parameter_gradient_analysis"]
    if protocol["dtype"] != "float64":
        parser.error("Figure-3 analysis requires float64")
    seeds = list(config["reproducibility"]["seeds"])
    past_contexts = list(protocol["past_contexts"])
    future_contexts = list(protocol["future_contexts"])
    update_length = int(config["optimization"]["update_length"])
    max_updates = int(config["optimization"]["max_updates"])
    device = resolve_device(args.device or config["reproducibility"]["device"])
    prepare_output_directory(ANALYSIS_DIRECTORY, overwrite=args.overwrite)

    data = load_wh(config["data"]["train_fraction"])
    starts = sample_update_starts(
        len(data.train.u), update_length=update_length, max_updates=int(protocol["interval_count"]),
        seed=int(protocol["sampling_seed"]), past_margin=max(past_contexts), future_margin=max(future_contexts),
    )
    errors: dict[tuple[int, int], list[float]] = {
        (past_context, future_context): []
        for future_context in future_contexts for past_context in past_contexts
    }
    checkpoint_identifiers: list[str] = []
    for seed in seeds:
        checkpoint = OUTPUT_ROOT / "bptt" / f"run_seed_{seed:03d}" / "final_model.pt"
        model, metadata = load_model(checkpoint, device=device, dtype=torch.float64)
        require_final_bptt_checkpoint(metadata, checkpoint, seed=seed, max_updates=max_updates)
        checkpoint_identifiers.append(str(checkpoint))
        (inputs, targets), = prepare_sequences(model, data.train)
        model.train()  # Required for cuDNN RNN backward; no stochastic layers are present.
        for start in starts:
            update_end = start + update_length - 1
            full_gradient = flatten_parameter_gradient(
                model, full_context_loss(model, inputs, targets, a=start, update_length=update_length)[0],
            ).detach()
            for future_context in future_contexts:
                for past_context in past_contexts:
                    objective, _ = tbptt_loss(
                        model,
                        inputs[start - past_context:update_end + future_context + 1],
                        targets[start - past_context:update_end + future_context + 1],
                        T_P=past_context, T_F=future_context, update_length=update_length,
                    )
                    truncated_gradient = flatten_parameter_gradient(model, objective).detach()
                    errors[past_context, future_context].append(
                        torch.linalg.vector_norm(full_gradient - truncated_gradient).item())
        print(f"Seed {seed} complete", flush=True)

    comparison_count = len(seeds) * len(starts)
    aggregate = [
        {"T_P": past_context, "T_F": future_context,
         "median_absolute_error": median(errors[past_context, future_context]),
         "sample_count": comparison_count}
        for future_context in future_contexts for past_context in past_contexts
    ]
    write_csv(ANALYSIS_DIRECTORY / "aggregated_gradient_errors.csv", aggregate)
    metadata = {
        "dtype": "float64",
        "contexts": {"past": past_contexts, "future": future_contexts},
        "update_length": update_length,
        "analysis_windows": [
            {"id": index, "update_start": start, "update_end": start + update_length - 1}
            for index, start in enumerate(starts)
        ],
        "checkpoint_identifiers": checkpoint_identifiers,
        "aggregation": f"median over {comparison_count} fixed-parameter gradient comparisons per context cell",
    }
    (ANALYSIS_DIRECTORY / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
