"""Train WH with generic one- or two-sided TBPTT on a shared update schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gradient_turnpike.data import load_wh
from gradient_turnpike.evaluation import load_model, save_evaluation
from gradient_turnpike.gradients import sample_update_starts
from gradient_turnpike.models import VanillaRNN
from gradient_turnpike.outputs import plot_history, read_resume_checkpoint, save_checkpoint, save_config, save_history
from gradient_turnpike.schedules import staged_learning_rate
from gradient_turnpike.tbptt import train_tbptt
from gradient_turnpike.utils import (
    load_config, prepare_output_directory, resolve_device, runtime_metadata,
    set_reproducibility, tbptt_run_directory,
)


DEFAULT_CONFIG = Path("configs/wh_paper.json")


def main() -> None:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_argument, _ = preliminary.parse_known_args()
    config = load_config(config_argument.config)
    optimization, model_settings = config["optimization"], config["model"]
    parser = argparse.ArgumentParser(description=__doc__, parents=[preliminary])
    parser.add_argument("--T-P", dest="T_P", type=int, default=20)
    parser.add_argument("--T-F", dest="T_F", type=int, default=20)
    parser.add_argument("--max-updates", type=int,
                        default=optimization["max_updates"],
                        help="Total planned optimizer updates.")
    parser.add_argument("--stop-after", type=int,
                        help="Stop this invocation at a global update; retain the max-updates schedule.")
    parser.add_argument("--validation-every-updates", type=int, default=optimization["validation_every_updates"])
    parser.add_argument("--update-length", type=int, default=optimization["update_length"])
    parser.add_argument("--state-dim", type=int, default=model_settings["state_dim"])
    parser.add_argument("--learning-rate", type=float, default=optimization["learning_rate"])
    parser.add_argument("--train-fraction", type=float, default=config["data"]["train_fraction"])
    parser.add_argument("--seed", type=int, default=config["reproducibility"]["seeds"][0])
    parser.add_argument("--run-id", default="001")
    parser.add_argument("--lr-schedule", choices=("constant", "staged"),
                        default=optimization["learning_rate_schedule"]["name"])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=config["reproducibility"]["device"])
    parser.add_argument("--dtype", choices=("float32", "float64"), default=optimization["dtype"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", type=Path,
                        help="Restore current model, Adam, schedule, and best-validation state.")
    parser.add_argument("--validation-segment-length", type=int, default=2000)
    args = parser.parse_args()
    if min(args.T_P, args.T_F) < 0 or min(args.max_updates, args.validation_every_updates,
                                           args.update_length, args.state_dim) < 1:
        parser.error("context lengths must be nonnegative; state and update values must be positive")
    output_dir = tbptt_run_directory(args.T_P, args.T_F, args.run_id)
    try:
        if args.resume is None:
            prepare_output_directory(output_dir, overwrite=args.overwrite)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
        device = resolve_device(args.device)
    except (FileExistsError, RuntimeError) as error:
        parser.error(str(error))
    set_reproducibility(args.seed)
    data = load_wh(args.train_fraction)
    margin = config["sampling"]["common_context_margin"]
    if args.T_P > margin or args.T_F > margin:
        parser.error("requested context exceeds the configured shared-schedule margin")
    update_schedule = sample_update_starts(len(data.train.u), update_length=args.update_length,
                                  max_updates=args.max_updates, seed=args.seed,
                                  past_margin=margin, future_margin=margin)
    (output_dir / "update_schedule.json").write_text(json.dumps(update_schedule) + "\n", encoding="utf-8")
    dtype = getattr(torch, args.dtype)
    model_config = {"input_size": data.train.u.shape[1], "hidden_size": args.state_dim,
                    "output_size": data.train.y.shape[1]}
    model = VanillaRNN(**model_config).to(device=device, dtype=dtype)
    metadata = {
        "experiment": config["experiment"], "resolved_configuration": config,
        "config_path": str(args.config), "training_method": "tbptt",
        "seed": args.seed, "device": device, "dtype": args.dtype, "output_dir": str(output_dir),
        "state_dimension": args.state_dim, "model_config": model_config, "optimizer": "Adam",
        "learning_rate": args.learning_rate, "T_P": args.T_P, "T_F": args.T_F,
        "learning_rate_schedule": args.lr_schedule,
        "update_length": args.update_length, "max_updates": args.max_updates,
        "validation_every_updates": args.validation_every_updates,
        "schedule": "deterministic shared update starts; full grid context margin", "schedule_margin": margin,
        "normalization": config["data"]["normalization"],
        "data_split": "chronological 80/20 development split; official test held out",
        "validation_metric": "free-running validation MSE after replaying training inputs",
        "train_samples": len(data.train.u),
        "validation_samples": len(data.validation.u), "sampling_time": data.train.sampling_time,
        "parameter_gradient_support": "parameter occurrences only in U=[a,b]",
        "context_rule": "fixed parameter values outside U; state derivatives remain connected through future context",
        **runtime_metadata(device),
    }
    resume_compatibility = {
        "method": "tbptt", "seed": args.seed, "dtype": args.dtype,
        "state_dimension": args.state_dim, "update_length": args.update_length,
        "past_context": args.T_P, "future_context": args.T_F,
        "max_updates": args.max_updates, "validation_every_updates": args.validation_every_updates,
        "learning_rate": args.learning_rate, "learning_rate_schedule": args.lr_schedule,
        "schedule_margin": margin, "train_fraction": args.train_fraction,
    }
    save_config(output_dir, metadata)

    def save_checkpoints(model_to_save: VanillaRNN, record: dict) -> None:
        if record["update"] == args.max_updates:
            save_checkpoint(output_dir, "final_model.pt", model_to_save,
                            {**metadata, "checkpoint_update": record["update"],
                             "checkpoint_validation_mse": record["val_loss"]})

    resume_boundaries = {1500, 3000, args.max_updates}
    if args.stop_after is not None:
        resume_boundaries.add(args.stop_after)
    history = train_tbptt(model, data.train, data.validation, T_P=args.T_P, T_F=args.T_F,
                          update_length=args.update_length, max_updates=args.max_updates,
                          validation_every_updates=args.validation_every_updates,
                          learning_rate=args.learning_rate,
                          learning_rate_schedule=staged_learning_rate if args.lr_schedule == "staged" else None,
                          update_starts=update_schedule, seed=args.seed,
                          on_update=save_checkpoints, resume_checkpoint=str(args.resume) if args.resume else None,
                          resume_output_directory=output_dir, resume_metadata=resume_compatibility,
                          resume_every_updates=tuple(sorted(resume_boundaries)),
                          stop_after=args.stop_after)
    completed_updates = args.stop_after if args.stop_after is not None else args.max_updates
    if completed_updates < args.max_updates:
        save_history(output_dir, history)
        save_config(output_dir, {**metadata, "completed_updates": completed_updates,
                                 "resume_checkpoint": str(output_dir / f"resume_update_{completed_updates:05d}.pt")})
        print(f"TBPTT T_P={args.T_P}, T_F={args.T_F}, seed={args.seed}: stopped after "
              f"global update {completed_updates}; resume from the saved raw-current-state checkpoint")
        return
    validation_records = [row for row in history if row["val_loss"] is not None]
    if args.resume is not None:
        resume_payload = read_resume_checkpoint(output_dir / f"resume_update_{args.max_updates:05d}.pt")
        best = {"update": resume_payload["best_validation_update"],
                "val_loss": resume_payload["best_validation_loss"]}
    else:
        best = min(validation_records, key=lambda row: row["val_loss"])
    metadata.update(best_validation_update=best["update"], best_validation_mse=best["val_loss"])
    checkpoint = save_checkpoint(output_dir, "best_model.pt", model,
                                 {**metadata, "checkpoint_update": best["update"],
                                  "checkpoint_validation_mse": best["val_loss"]})
    save_history(output_dir, history)
    plot_history(output_dir, history)
    best_model, _ = load_model(checkpoint, device=device, dtype=dtype)
    metadata["evaluation"] = save_evaluation(output_dir, best_model, data,
                                              segment_length=args.validation_segment_length)
    metadata["checkpoint_path"] = str(checkpoint)
    save_config(output_dir, metadata)
    print(f"TBPTT T_P={args.T_P}, T_F={args.T_F}, seed={args.seed}: "
          f"best validation MSE={best['val_loss']:.6g} at update {best['update']}")


if __name__ == "__main__":
    main()
