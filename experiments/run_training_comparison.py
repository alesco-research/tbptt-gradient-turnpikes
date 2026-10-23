"""Run Table I: validation-selected BPTT and symmetric TBPTT at a fixed budget.

TBPTT contexts share initialization and sampling for each seed. Test data are
scored only after checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from gradient_turnpike.data import WHData, load_wh
from gradient_turnpike.evaluation import (
    evaluate_wh_test_metrics,
    evaluate_wh_training_metrics,
)
from gradient_turnpike.gradients import sample_update_minibatches, sample_update_starts
from gradient_turnpike.metrics import augment_paired_bptt_metrics, summarize_training_comparison
from gradient_turnpike.models import VanillaRNN
from gradient_turnpike.outputs import save_checkpoint
from gradient_turnpike.schedules import staged_learning_rate
from gradient_turnpike.tbptt import train_tbptt
from gradient_turnpike.training import train_bptt
from gradient_turnpike.utils import load_config, resolve_device, set_reproducibility


ROOT = Path("outputs/analysis/training_comparison")
DEFAULT_CONFIG = Path("configs/wh_paper.json")


def paper_configuration(base: dict) -> dict:
    """The frozen scientific protocol, independent of execution mode."""
    source = base["training_comparison"]
    bptt = source["bptt"]
    tbptt = source["tbptt"]
    validation_selection = source["validation_selection"]
    return {
        "experiment": "wh_training_comparison_final",
        "seeds": list(base["reproducibility"]["seeds"]),
        "state_dimension": base["model"]["state_dim"],
        "dtype": "float32",
        "device": base["reproducibility"]["device"],
        "validation_selection": validation_selection,
        "bptt": {"update_length": bptt["update_length"], "mini_batch_size": bptt["mini_batch_size"],
                  "max_updates": bptt["max_updates"], "sampling_seed": bptt["sampling_seed"],
                  "past_context_margin": bptt["past_context_margin"],
                  "future_context_margin": bptt["future_context_margin"],
                  "full_context": bptt["full_context"]},
        "tbptt": {"contexts": list(tbptt["symmetric_contexts"]),
                   "update_length": tbptt["update_length"], "mini_batch_size": tbptt["mini_batch_size"],
                   "max_updates": tbptt["max_updates"], "sampling_seed": tbptt["sampling_seed"],
                   "past_context_margin": tbptt["past_context_margin"],
                   "future_context_margin": tbptt["future_context_margin"],
                   "start_domain": tbptt["common_update_start_domain"], "symmetric": True},
        "optimizer": {"name": "Adam", "learning_rate_schedule": {
            "updates_1_1500": 1e-3, "updates_1501_3000": 3e-4,
            "updates_3001_5000": 1e-4}},
        "test_usage": "final evaluation only after frozen training/model-selection protocol",
        "runtime_measurement": {
            "definition": "synchronized optimizer-update computation only; validation, checkpointing, and final evaluation excluded",
            "cuda_synchronization": "immediately before and after every optimizer update",
            "provenance_fields": ["model_metrics_source", "timing_source"],
        },
    }


def _model(data: WHData, device: str, *, state_dimension: int) -> VanillaRNN:
    """Create the paper RNN from the resolved public configuration."""
    return VanillaRNN(input_size=data.train.u.shape[1], hidden_size=state_dimension,
                      output_size=data.train.y.shape[1]).to(device=device, dtype=torch.float32)


def _best_record(history: list[dict]) -> dict:
    return min((row for row in history if row["val_loss"] is not None), key=lambda row: row["val_loss"])


def _row(*, seed: int, method: str, context: int | None, update_length: int,
          mini_batch_size: int, best_update: int, best_validation_mse: float,
          model: VanillaRNN, data: WHData, training_time_seconds: float,
          model_metrics_source: str, timing_source: str, include_test: bool = True) -> dict:
    full_training_mse, full_training_rmse = evaluate_wh_training_metrics(model, data.train)
    test_mse, test_rmse = (evaluate_wh_test_metrics(model, data.test, data.official_test)
                           if include_test else (float("nan"), float("nan")))
    return {
        "seed": seed, "method": method, "T_P": "full" if context is None else context,
        "T_F": "full" if context is None else context, "update_length": update_length,
        "mini_batch_size": mini_batch_size, "model_metrics_source": model_metrics_source,
        "best_validation_update": best_update, "best_validation_mse": best_validation_mse,
        "full_training_mse": full_training_mse, "full_training_rmse": full_training_rmse,
        "test_mse": test_mse, "test_rmse": test_rmse,
        "training_time_seconds": training_time_seconds,
        "timing_source": timing_source,
    }


def _train_bptt(seed: int, data: WHData, device: str, config: dict, *, max_updates: int,
                output_root: Path, include_test: bool = True) -> dict:
    set_reproducibility(seed)
    model = _model(data, device, state_dimension=int(config["state_dimension"]))
    spec = config["bptt"]
    starts = sample_update_starts(len(data.train.u), update_length=spec["update_length"],
                                  max_updates=max_updates, seed=seed,
                                  past_margin=spec["past_context_margin"], future_margin=spec["future_context_margin"])
    history = train_bptt(model, data.train, data.validation, update_length=spec["update_length"],
                         max_updates=max_updates, validation_every_updates=config["validation_selection"]["cadence_updates"],
                         learning_rate_schedule=staged_learning_rate, update_starts=starts, seed=seed,
                         verbose=True)
    best = _best_record(history)
    run_dir = output_root / f"bptt_seed_{seed:03d}"
    save_checkpoint(run_dir, "best_model.pt", model, {"model_config": {"input_size": 1,
                                                        "hidden_size": config["state_dimension"], "output_size": 1},
                                                        "seed": seed, "checkpoint_update": best["update"]})
    return _row(seed=seed, method="BPTT", context=None, update_length=spec["update_length"],
                mini_batch_size=1, best_update=best["update"], best_validation_mse=best["val_loss"],
                model=model, data=data, training_time_seconds=history[-1]["training_compute_seconds"],
                model_metrics_source="production_training_run",
                timing_source="synchronized_production_training_compute",
                include_test=include_test)


def _train_tbptt(seed: int, context: int, data: WHData, device: str, config: dict, *,
                 max_updates: int, output_root: Path, include_test: bool = True) -> dict:
    set_reproducibility(seed)  # identical initialization for every context of this seed
    model = _model(data, device, state_dimension=int(config["state_dimension"]))
    spec = config["tbptt"]
    schedule = sample_update_minibatches(
        len(data.train.u), update_length=spec["update_length"], max_updates=max_updates,
        mini_batch_size=spec["mini_batch_size"], seed=10_000 + seed,
        past_margin=spec["past_context_margin"], future_margin=spec["future_context_margin"])
    history = train_tbptt(model, data.train, data.validation, T_P=context, T_F=context,
                          update_length=spec["update_length"], max_updates=max_updates,
                          validation_every_updates=config["validation_selection"]["cadence_updates"],
                          learning_rate_schedule=staged_learning_rate, update_starts=schedule,
                          seed=seed, verbose=True)
    best = _best_record(history)
    run_dir = output_root / f"tbptt_T{context:02d}_seed_{seed:03d}"
    save_checkpoint(run_dir, "best_model.pt", model, {"model_config": {"input_size": 1,
                                                        "hidden_size": config["state_dimension"], "output_size": 1},
                                                        "seed": seed, "T_P": context, "T_F": context,
                                                        "checkpoint_update": best["update"]})
    return _row(seed=seed, method="TBPTT", context=context, update_length=spec["update_length"],
                mini_batch_size=spec["mini_batch_size"], best_update=best["update"],
                best_validation_mse=best["val_loss"], model=model, data=data,
                training_time_seconds=history[-1]["training_compute_seconds"],
                model_metrics_source="production_training_run",
                timing_source="synchronized_production_training_compute",
                include_test=include_test)


def write_outputs(root: Path, config: dict, rows: list[dict]) -> None:
    """Write the compact data required to reconstruct the paper table."""
    root.mkdir(parents=True, exist_ok=True)
    augment_paired_bptt_metrics(rows, expected_seeds=config["seeds"])
    summary, _ = summarize_training_comparison(rows)
    for name, values in (("per_run_metrics.csv", rows), ("summary.csv", summary)):
        with (root / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--paper", action="store_true", help="Run the frozen 25-model paper comparison.")
    mode.add_argument("--smoke", action="store_true", help="Run a five-update CPU/CUDA code-path smoke check.")
    parser.add_argument("--seed", type=int, default=1, help="Seed for --smoke only.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--overwrite", action="store_true", help="Replace compact final summary files.")
    args = parser.parse_args()
    base = load_config(args.config)
    config = paper_configuration(base)
    device = resolve_device(args.device or config["device"])
    config["resolved_device"] = device
    data = load_wh(base["data"]["train_fraction"])
    if args.smoke:
        seed = args.seed
        smoke_config = {**config, "bptt": {**config["bptt"]}, "tbptt": {**config["tbptt"]}, "mode": "smoke"}
        for block in (smoke_config["bptt"], smoke_config["tbptt"]):
            block["max_updates"] = 5
        rows = [_train_bptt(seed, data, device, smoke_config, max_updates=5,
                            output_root=Path("outputs/audit/training_comparison_smoke"), include_test=False)]
        rows += [_train_tbptt(seed, context, data, device, smoke_config, max_updates=5,
                              output_root=Path("outputs/audit/training_comparison_smoke"), include_test=False)
                 for context in config["tbptt"]["contexts"]]
        print(json.dumps(rows, indent=2))
        return
    if ROOT.exists() and any(ROOT.iterdir()) and not args.overwrite:
        parser.error(f"{ROOT} exists; use --overwrite after reviewing the saved final comparison")
    rows: list[dict] = []
    run_root = Path("outputs/bptt/training_comparison_final")
    for seed in config["seeds"]:
        rows.append(_train_bptt(seed, data, device, config,
                                max_updates=config["bptt"]["max_updates"], output_root=run_root))
    run_root = Path("outputs/tbptt/training_comparison_final")
    for seed in config["seeds"]:
        for context in config["tbptt"]["contexts"]:
            rows.append(_train_tbptt(seed, context, data, device, config,
                                     max_updates=config["tbptt"]["max_updates"], output_root=run_root))
    expected_records = len(config["seeds"]) * (1 + len(config["tbptt"]["contexts"]))
    if len(rows) != expected_records:
        raise RuntimeError(f"Expected {expected_records} final records, found {len(rows)}")
    write_outputs(ROOT, config, rows)
    print(f"Wrote {len(rows)} records to {ROOT}")


if __name__ == "__main__":
    main()
