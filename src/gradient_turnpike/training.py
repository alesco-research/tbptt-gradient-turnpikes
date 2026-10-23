"""Full-context update-interval BPTT with PyTorch autograd and Adam."""

import math
import time
from collections.abc import Callable

import torch
from torch import nn

from gradient_turnpike.data import SignalData
from gradient_turnpike.gradients import full_context_loss, sample_update_starts
from gradient_turnpike.models import VanillaRNN
from gradient_turnpike.simulation import simulate
from gradient_turnpike.outputs import load_resume_checkpoint, resume_bookkeeping, save_resume_checkpoint


def prepare_sequences(model: VanillaRNN, *sequences: SignalData) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Validate signals and move them to the model's device/dtype."""
    parameter = next(model.parameters())
    prepared = []
    for data in sequences:
        if (data.u.ndim != 2 or data.y.ndim != 2 or len(data.u) == 0
                or len(data.u) != len(data.y)
                or data.u.shape[1] != model.rnn.input_size
                or data.y.shape[1] != model.readout.out_features):
            raise ValueError("Signals must have matching nonempty time and model channel dimensions")
        u, y = (signal.to(device=parameter.device, dtype=parameter.dtype)
                for signal in (data.u, data.y))
        if not torch.isfinite(u).all() or not torch.isfinite(y).all():
            raise ValueError("Training and validation signals must be finite")
        prepared.append((u, y))
    return prepared


@torch.no_grad()
def evaluate_validation(model: VanillaRNN, u_train: torch.Tensor,
                        u_val: torch.Tensor, y_val: torch.Tensor) -> float:
    """Free-running MSE after replaying training inputs; leaves eval mode set."""
    model.eval()
    _, train_states = simulate(model, u_train)
    prediction, _ = simulate(model, u_val, train_states[-1])
    return nn.functional.mse_loss(prediction, y_val, reduction="mean").item()


def train_bptt(
    model: VanillaRNN,
    train: SignalData,
    validation: SignalData,
    *,
    update_length: int,
    max_updates: int,
    validation_every_updates: int,
    learning_rate: float = 1e-3,
    learning_rate_schedule: Callable[[int], float] | None = None,
    update_starts: list[int] | None = None,
    seed: int = 0,
    verbose: bool = True,
    on_update: Callable[[VanillaRNN, dict], None] | None = None,
    resume_checkpoint: str | None = None,
    resume_output_directory=None,
    resume_metadata: dict | None = None,
    resume_every_updates: tuple[int, ...] = (),
    stop_after: int | None = None,
) -> list[dict]:
    """Fit with full-context G_U gradients and periodic free-running validation.

    max_updates fixes the schedule; stop_after bounds only this invocation.
    Resume restores the current model/Adam state. At completion, restore the
    strictly best validation model, retaining the current state in checkpoints.
    """
    if min(update_length, max_updates, validation_every_updates) < 1:
        raise ValueError("update length, update budget, and validation interval must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    (u_train, y_train), (u_val, y_val) = prepare_sequences(model, train, validation)
    starts = update_starts or sample_update_starts(
        len(u_train), update_length=update_length, max_updates=max_updates, seed=seed)
    if len(starts) != max_updates or any(a < 0 or a + update_length > len(u_train) for a in starts):
        raise ValueError("update_starts must contain one complete interval per optimizer update")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict] = []
    completed = 0
    best_loss, best_state, best_update = math.inf, None, None
    training_compute_seconds = 0.0
    active_wall_clock_seconds = 0.0
    if resume_checkpoint is not None:
        payload = load_resume_checkpoint(resume_checkpoint, model, optimizer)
        if payload["schedule"] != starts or payload["metadata"].get("update_length") != update_length:
            raise ValueError("resume checkpoint is incompatible with the requested schedule or update length")
        if resume_metadata is not None and payload["metadata"] != resume_metadata:
            raise ValueError("resume checkpoint is incompatible with the requested scientific configuration")
        completed = int(payload["completed_updates"])
        best_loss = payload.get("best_validation_loss", math.inf)
        best_state = payload.get("best_model_state_dict")
        best_update = payload.get("best_validation_update")
        history, training_compute_seconds, active_wall_clock_seconds = resume_bookkeeping(payload)
        if not 0 <= completed < max_updates:
            raise ValueError("resume completed_updates is outside the requested training range")
    stop_after = max_updates if stop_after is None else stop_after
    if not completed <= stop_after <= max_updates:
        raise ValueError("stop_after must be between completed_updates and max_updates")
    start_time = time.perf_counter()
    for update, a in enumerate(starts[completed:stop_after], start=completed + 1):
        model.train()
        active_learning_rate = learning_rate_schedule(update) if learning_rate_schedule else learning_rate
        for group in optimizer.param_groups:
            group["lr"] = active_learning_rate
        # Time synchronized updates only; exclude validation, callbacks, and I/O.
        if model.rnn.weight_hh_l0.is_cuda:
            torch.cuda.synchronize(model.rnn.weight_hh_l0.device)
        compute_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        objective, update_loss = full_context_loss(
            model, u_train, y_train, a=a, update_length=update_length)
        if not torch.isfinite(objective):
            raise FloatingPointError(f"Nonfinite full-context objective at update {update}")
        objective.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            raise FloatingPointError(f"Nonfinite full-context gradient at update {update}")
        optimizer.step()
        if model.rnn.weight_hh_l0.is_cuda:
            torch.cuda.synchronize(model.rnn.weight_hh_l0.device)
        training_compute_seconds += time.perf_counter() - compute_start
        val_loss = None
        if update % validation_every_updates == 0 or update == max_updates:
            val_loss = evaluate_validation(model, u_train, u_val, y_val)
            if not math.isfinite(val_loss):
                raise FloatingPointError(f"Nonfinite validation loss at update {update}")
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {name: value.detach().cpu().clone()
                               for name, value in model.state_dict().items()}
                best_update = update
        record = {"update": update, "update_start": a, "learning_rate": active_learning_rate,
                  "train_loss": update_loss.item(),
                  "val_loss": val_loss, "wall_clock_time": active_wall_clock_seconds + time.perf_counter() - start_time,
                  "training_compute_seconds": training_compute_seconds,
                  "processed_update_samples": update * update_length,
                  "processed_context_samples": update * len(u_train)}
        history.append(record)
        if on_update is not None:
            on_update(model, record)
        if resume_output_directory is not None and update in resume_every_updates:
            save_resume_checkpoint(resume_output_directory / f"resume_update_{update:05d}.pt", model, optimizer,
                                    completed_updates=update, schedule=starts,
                                   metadata=resume_metadata or {"update_length": update_length},
                                   best_model_state=best_state, best_validation_loss=best_loss,
                                   best_validation_update=best_update, history=history,
                                   training_compute_seconds=training_compute_seconds,
                                   active_wall_clock_seconds=record["wall_clock_time"])
        if verbose and val_loss is not None:
            print(f"Update {update:5d}: train MSE={record['train_loss']:.8g}, "
                  f"validation MSE={val_loss:.8g}", flush=True)
    if stop_after == max_updates:
        model.load_state_dict(best_state)
    return history
