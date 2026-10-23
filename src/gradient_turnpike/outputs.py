"""Output helpers shared by training experiments, independent of the method."""

import csv
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from gradient_turnpike.plotting import save_figure


def save_checkpoint(output_dir: Path, filename: str, model: nn.Module, metadata: dict) -> Path:
    """Save portable model weights and metadata; this is not a resume checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    torch.save({"model_state_dict": {k: v.detach().cpu().clone()
                                     for k, v in model.state_dict().items()},
                "metadata": metadata}, path)
    return path


def save_resume_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, *,
                           completed_updates: int, schedule: list, metadata: dict,
                           best_model_state: dict | None = None, best_validation_loss: float | None = None,
                           best_validation_update: int | None = None, history: list[dict],
                           training_compute_seconds: float, active_wall_clock_seconds: float) -> Path:
    """Save model, Adam, RNG, best-model, and history state for exact resumption.

    active_wall_clock_seconds excludes pauses between trainer invocations.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "completed_updates": completed_updates, "schedule": schedule, "metadata": metadata,
                "best_model_state_dict": best_model_state, "best_validation_loss": best_validation_loss,
                "best_validation_update": best_validation_update,
                "history": history, "training_compute_seconds": training_compute_seconds,
                "active_wall_clock_seconds": active_wall_clock_seconds,
                "python_random_state": random.getstate(),
                "torch_cpu_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, path)
    return path


def read_resume_checkpoint(path: Path) -> dict:
    """Read resumable-training metadata without mutating a model or optimizer."""
    return torch.load(path, map_location="cpu", weights_only=False)


def load_resume_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    """Restore model, Adam, and RNG state; return saved logical-run state."""
    payload = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    random.setstate(payload["python_random_state"])
    torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
    if payload["torch_cuda_rng_state_all"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["torch_cuda_rng_state_all"]])
    return payload


def resume_bookkeeping(payload: dict) -> tuple[list[dict], float, float]:
    """Recover accumulated history/timing; reject incomplete resume records."""
    required = ("history", "training_compute_seconds", "active_wall_clock_seconds", "python_random_state")
    if any(key not in payload for key in required):
        raise ValueError("resume checkpoint lacks complete history/timing bookkeeping")
    completed = int(payload["completed_updates"])
    history = list(payload["history"])
    if [record.get("update") for record in history] != list(range(1, completed + 1)):
        raise ValueError("resume checkpoint history does not match completed_updates")
    training_compute_seconds = float(payload["training_compute_seconds"])
    active_wall_clock_seconds = float(payload["active_wall_clock_seconds"])
    if (not math.isfinite(training_compute_seconds) or not math.isfinite(active_wall_clock_seconds)
            or training_compute_seconds < 0 or active_wall_clock_seconds < 0):
        raise ValueError("resume checkpoint has invalid accumulated timing")
    if history and (history[-1].get("training_compute_seconds") != training_compute_seconds
                    or history[-1].get("wall_clock_time") != active_wall_clock_seconds):
        raise ValueError("resume checkpoint timing does not match its final history record")
    return history, training_compute_seconds, active_wall_clock_seconds


def save_config(output_dir: Path, config: dict) -> None:
    """Save both the compatibility and explicit resolved-config filenames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(config, indent=2) + "\n"
    (output_dir / "config.json").write_text(payload, encoding="utf-8")
    (output_dir / "resolved_config.json").write_text(payload, encoding="utf-8")


def save_history(output_dir: Path, history: list[dict]) -> None:
    """Save update-indexed losses and method-independent progress counters."""
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["update", "train_loss", "val_loss", "wall_clock_time", "training_compute_seconds",
               "processed_update_samples", "processed_context_samples"]
    for record in history:
        columns.extend(key for key in record if key not in columns)
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


def plot_history(output_dir: Path, history: list[dict]) -> None:
    """Plot update losses and periodic validation losses versus optimizer update."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    updates = [row["update"] for row in history]
    ax.plot(updates, [row["train_loss"] for row in history], linewidth=.8, label="Update MSE")
    validation = [row for row in history if row["val_loss"] is not None]
    ax.plot([row["update"] for row in validation], [row["val_loss"] for row in validation],
            marker="o", markersize=3, linewidth=1.1, label="Free-running validation MSE")
    ax.set(xlabel="Optimizer update", ylabel="Mean squared error", yscale="log")
    ax.legend()
    save_figure(fig, output_dir / "validation_loss_vs_updates")
