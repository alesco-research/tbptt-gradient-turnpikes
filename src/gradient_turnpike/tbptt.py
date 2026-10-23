"""One generic trainer for past-only and two-sided TBPTT context."""

from collections.abc import Callable
import math
import time

import torch

from gradient_turnpike.data import SignalData
from gradient_turnpike.gradients import (
    ContextWindow,
    minibatch_tbptt_loss,
    make_context_window,
    sample_update_starts,
    tbptt_loss,
)
from gradient_turnpike.models import VanillaRNN
from gradient_turnpike.outputs import load_resume_checkpoint, resume_bookkeeping, save_resume_checkpoint
from gradient_turnpike.training import evaluate_validation, prepare_sequences


def train_tbptt(
    model: VanillaRNN, train: SignalData, validation: SignalData, *,
    T_P: int, T_F: int, update_length: int, max_updates: int,
    validation_every_updates: int, learning_rate: float = 1e-3,
    learning_rate_schedule: Callable[[int], float] | None = None,
    update_starts: list[int] | None = None, seed: int = 0,
    verbose: bool = True,
    on_update: Callable[[VanillaRNN, dict], None] | None = None,
    resume_checkpoint: str | None = None,
    resume_output_directory=None,
    resume_metadata: dict | None = None,
    resume_every_updates: tuple[int, ...] = (),
    stop_after: int | None = None,
) -> list[dict]:
    """Fit with the mean G_hat_U over independently initialized contexts.

    Each batch retains duplicate intervals. As in train_bptt, max_updates fixes
    the schedule, resume restores the current model/Adam state, and completion
    restores the strictly best free-running validation model.
    """
    if min(T_P, T_F) < 0 or min(update_length, max_updates, validation_every_updates) < 1:
        raise ValueError("contexts must be nonnegative; update values must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    (u_train, y_train), (u_val, y_val) = prepare_sequences(model, train, validation)
    schedule = update_starts or sample_update_starts(
        len(u_train), update_length=update_length, max_updates=max_updates, seed=seed,
        past_margin=T_P, future_margin=T_F)
    if (len(schedule) != max_updates or any(
            make_context_window(a, update_length, T_P, T_F).context_start < 0
            or make_context_window(a, update_length, T_P, T_F).context_end >= len(u_train)
            for item in schedule for a in (item if isinstance(item, list) else [item]))):
        raise ValueError("update_starts must contain complete TBPTT contexts")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict] = []
    completed = 0
    best_loss, best_state, best_update = math.inf, None, None
    training_compute_seconds = 0.0
    active_wall_clock_seconds = 0.0
    if resume_checkpoint is not None:
        payload = load_resume_checkpoint(resume_checkpoint, model, optimizer)
        if payload["schedule"] != schedule:
            raise ValueError("resume checkpoint is incompatible with the requested schedule")
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
    for update, scheduled in enumerate(schedule[completed:stop_after], start=completed + 1):
        batch = scheduled if isinstance(scheduled, list) else [scheduled]
        a = batch[0]
        model.train()
        active_learning_rate = learning_rate_schedule(update) if learning_rate_schedule else learning_rate
        for group in optimizer.param_groups:
            group["lr"] = active_learning_rate
        if model.rnn.weight_hh_l0.is_cuda:
            torch.cuda.synchronize(model.rnn.weight_hh_l0.device)
        compute_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if len(batch) == 1:
            context_window: ContextWindow = make_context_window(a, update_length, T_P, T_F)
            objective, update_loss = tbptt_loss(model, u_train[context_window.context_slice], y_train[context_window.context_slice],
                                                T_P=T_P, T_F=T_F, update_length=update_length)
        else:
            objective, update_loss = minibatch_tbptt_loss(
                model, u_train, y_train, starts=batch, T_P=T_P, T_F=T_F, update_length=update_length)
        if not torch.isfinite(objective):
            raise FloatingPointError(f"Nonfinite objective at update {update}, update starts={batch}")
        objective.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            raise FloatingPointError(f"Nonfinite gradient at update {update}, update starts={batch}")
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
        record = {"update": update, "update_start": a, "mini_batch_starts": batch, "learning_rate": active_learning_rate,
                  "train_loss": update_loss.item(),
                  "val_loss": val_loss, "gradient_objective": objective.item(),
                  "wall_clock_time": active_wall_clock_seconds + time.perf_counter() - start_time,
                  "training_compute_seconds": training_compute_seconds,
                  "processed_update_samples": update * update_length,
                  "processed_context_samples": update * (T_P + update_length + T_F)}
        history.append(record)
        if on_update is not None:
            on_update(model, record)
        if resume_output_directory is not None and update in resume_every_updates:
            save_resume_checkpoint(resume_output_directory / f"resume_update_{update:05d}.pt", model, optimizer,
                                   completed_updates=update, schedule=schedule,
                                   metadata=resume_metadata or {"T_P": T_P, "T_F": T_F,
                                                               "update_length": update_length},
                                   best_model_state=best_state, best_validation_loss=best_loss,
                                   best_validation_update=best_update, history=history,
                                   training_compute_seconds=training_compute_seconds,
                                   active_wall_clock_seconds=record["wall_clock_time"])
        if verbose and val_loss is not None:
            print(f"Update {update:5d}: update MSE={record['train_loss']:.8g}, "
                  f"validation MSE={val_loss:.8g}", flush=True)
    if stop_after == max_updates:
        model.load_state_dict(best_state)
    return history
