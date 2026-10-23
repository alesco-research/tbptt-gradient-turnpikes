"""Parameter-occurrence-restricted gradients for context windows."""

from dataclasses import dataclass
import random

import torch
from torch import nn
from torch.func import functional_call

from gradient_turnpike.models import VanillaRNN
from gradient_turnpike.simulation import simulate


@dataclass(frozen=True)
class ContextWindow:
    """Inclusive U=[a,b] and C=[a-T_P,b+T_F]; use context_slice for Python."""

    context_start: int
    update_start: int
    update_end: int
    context_end: int

    @property
    def update_length(self) -> int:
        """Number of samples in the inclusive update interval ``U``."""
        return self.update_end - self.update_start + 1

    @property
    def context_length(self) -> int:
        """Number of samples in the inclusive context interval ``C``."""
        return self.context_end - self.context_start + 1

    @property
    def context_slice(self) -> slice:
        """Python slice selecting the complete context ``C``."""
        return slice(self.context_start, self.context_end + 1)


def make_context_window(
    update_start: int, update_length: int, past_context: int, future_context: int,
) -> ContextWindow:
    """Construct inclusive U and C; trajectory bounds are checked separately."""
    if min(update_start, past_context, future_context) < 0 or update_length < 1:
        raise ValueError("update_start and context extensions must be nonnegative; update_length positive")
    update_end = update_start + update_length - 1
    return ContextWindow(update_start - past_context, update_start, update_end,
                         update_end + future_context)


def validate_context_window(window: ContextWindow, trajectory_length: int) -> None:
    """Raise when an inclusive context interval does not fit a trajectory."""
    if trajectory_length < 1 or not (0 <= window.context_start <= window.update_start
                                     <= window.update_end <= window.context_end < trajectory_length):
        raise ValueError("context window must fit the finite trajectory and contain its update interval")


def admissible_update_start_bounds(
    trajectory_length: int, *, update_length: int, past_context: int = 0,
    future_context: int = 0,
) -> tuple[int, int]:
    """Return inclusive valid starts for complete ``U`` and ``C`` intervals."""
    if trajectory_length < 1 or update_length < 1 or min(past_context, future_context) < 0:
        raise ValueError("trajectory/update lengths must be positive; context extensions nonnegative")
    first = past_context
    last = trajectory_length - future_context - update_length
    if first > last:
        raise ValueError("No update interval fits the requested context extensions")
    return first, last


def sample_update_starts(
    n: int, *, update_length: int, max_updates: int, seed: int,
    past_margin: int = 0, future_margin: int = 0,
) -> list[int]:
    """Sample starts with replacement; common margins permit shared schedules."""
    if max_updates < 1:
        raise ValueError("max_updates must be positive")
    first, last = admissible_update_start_bounds(
        n, update_length=update_length, past_context=past_margin, future_context=future_margin)
    generator = random.Random(seed)
    return [generator.randint(first, last) for _ in range(max_updates)]


def sample_update_minibatches(
    n: int, *, update_length: int, max_updates: int, mini_batch_size: int, seed: int,
    past_margin: int = 0, future_margin: int = 0,
) -> list[list[int]]:
    """Group independently seeded starts into batches, preserving duplicates."""
    if mini_batch_size < 1:
        raise ValueError("mini_batch_size must be positive")
    flat = sample_update_starts(n, update_length=update_length,
                                max_updates=max_updates * mini_batch_size, seed=seed,
                                past_margin=past_margin, future_margin=future_margin)
    return [flat[i:i + mini_batch_size] for i in range(0, len(flat), mini_batch_size)]


def make_disjoint_context_windows(
    trajectory_length: int, past_context: int, future_context: int, update_length: int,
) -> list[ContextWindow]:
    """Disjoint updates with exact context; omit incomplete final intervals."""
    if any(not isinstance(value, int) for value in
           (trajectory_length, past_context, future_context, update_length)):
        raise ValueError("context-window dimensions must be integers")
    if trajectory_length < 1 or past_context < 0 or future_context < 0 or update_length < 1:
        raise ValueError("Require positive trajectory/update lengths and nonnegative contexts")
    windows = [make_context_window(a, update_length, past_context, future_context)
               for a in range(past_context, trajectory_length - future_context - update_length + 1,
                              update_length)]
    if not windows:
        raise ValueError("No complete context/update window fits the training sequence")
    return windows


class _Simulation(nn.Module):
    """Make the existing simulator callable with substituted model parameters."""

    def __init__(self, model: VanillaRNN):
        super().__init__()
        self.model = model

    def forward(self, u, x0):
        return simulate(self.model, u, x0)


def tbptt_loss(
    model: VanillaRNN, u: torch.Tensor, y: torch.Tensor, *,
    T_P: int, T_F: int, update_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a gradient surrogate and update MSE for exactly one context C.

    Signals must match the model's device/dtype. With x_r=0 and p_{s+1}=0,
    J = sum_{t=a}^s mean_channels((y_hat_t-y_t)**2) / |U|. Only parameter
    occurrences in U are active; future losses differentiate through the state
    at b+1 with outside-U parameters frozen. Thus grad(J) is G_hat_U, not the
    ordinary gradient of losses restricted to U. The second value logs MSE on U.
    Use train mode for cuDNN backward.
    """
    expected_length = T_P + update_length + T_F
    make_disjoint_context_windows(expected_length, T_P, T_F, update_length)
    if (u.ndim != 2 or y.ndim != 2 or len(u) != expected_length
            or len(y) != expected_length or u.shape[1] != model.rnn.input_size
            or y.shape[1] != model.readout.out_features):
        raise ValueError("Signals must match the exact context length and model channels")
    simulation = _Simulation(model)
    # Only parameter VALUES are detached, never the state at the future boundary.
    frozen = {name: parameter.detach() for name, parameter in simulation.named_parameters()}
    x = u.new_zeros(model.rnn.hidden_size)
    if T_P:
        # The entering state has no dependence on parameter occurrences in U.
        with torch.no_grad():
            _, states = functional_call(simulation, frozen, (u[:T_P], x), strict=True)
            x = states[-1]
    end = T_P + update_length
    prediction, states = simulate(model, u[T_P:end], x)
    update_mse = (prediction - y[T_P:end]).square().mean()
    objective = update_mse
    if T_F:
        future_prediction, _ = functional_call(
            simulation, frozen, (u[end:], states[-1]), strict=True)
        objective = objective + (future_prediction - y[end:]).square().sum() / (
            update_length * y.shape[1])
    return objective, update_mse


def full_context_loss(
    model: VanillaRNN, u: torch.Tensor, y: torch.Tensor, *, a: int, update_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a surrogate for G_U and update MSE, using the whole trajectory.

    Only occurrences in U are active; the full prefix supplies the state and
    the full suffix supplies the adjoint, as in tbptt_loss.
    """
    b = a + update_length - 1
    if not 0 <= a <= b < len(u):
        raise ValueError("The update interval must fit in the supplied trajectory")
    return tbptt_loss(model, u, y, T_P=a, T_F=len(u) - 1 - b,
                       update_length=update_length)


def minibatch_tbptt_loss(model: VanillaRNN, u: torch.Tensor, y: torch.Tensor, *, starts: list[int],
                         T_P: int, T_F: int, update_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact mean of independent existing TBPTT interval objectives."""
    values = [tbptt_loss(model, u[a-T_P:a+update_length+T_F], y[a-T_P:a+update_length+T_F],
                         T_P=T_P, T_F=T_F, update_length=update_length) for a in starts]
    return torch.stack([item[0] for item in values]).mean(), torch.stack([item[1] for item in values]).mean()


def flatten_parameter_gradient(model: VanillaRNN, objective: torch.Tensor) -> torch.Tensor:
    """Return one objective gradient in the model's stable parameter order."""
    values = torch.autograd.grad(objective, tuple(model.parameters()), allow_unused=True)
    return torch.cat([(torch.zeros_like(parameter) if value is None else value).reshape(-1)
                      for parameter, value in zip(model.parameters(), values)])


def _stage_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The per-time MSE whose mean across U is used by tbptt_loss."""
    return (prediction - target).square().mean()


def pointwise_gradient_contributions(
    model: VanillaRNN, u: torch.Tensor, y: torch.Tensor, *, a: int, b: int
) -> list[torch.Tensor]:
    """Return D_theta ell_t + D_theta f_t^T p_{t+1} for t in inclusive U=[a,b].

    The supplied context starts at x_0=0 and ends with p_N=0. Adjoint VJPs
    include every future loss in that context. Stored states have no graph;
    only one local VJP graph is live at a time. Returned vectors are unscaled
    parameter-occurrence contributions, not gradients of individual losses.
    """
    if (u.ndim != 2 or y.ndim != 2 or len(u) != len(y) or not len(u)
            or not 0 <= a <= b < len(u)
            or u.shape[1] != model.rnn.input_size
            or y.shape[1] != model.readout.out_features):
        raise ValueError("Context and update interval must match the model")
    model.train()  # cuDNN RNN backward requires training mode.
    frozen = {name: parameter.detach() for name, parameter in model.named_parameters()}
    with torch.no_grad():
        _, states = simulate(model, u)

    next_adjoint = u.new_zeros(model.rnn.hidden_size)
    adjoint_after = [None] * (b - a + 1)
    # Earlier adjoints cannot affect contributions in U.
    for time_index in range(len(u) - 1, a - 1, -1):
        state_t = states[time_index].detach().requires_grad_()
        next_state, prediction = functional_call(
            model, frozen, (state_t, u[time_index]), strict=True)
        local_lagrangian = _stage_loss(prediction, y[time_index]) + torch.dot(next_state, next_adjoint)
        if time_index <= b:
            adjoint_after[time_index - a] = next_adjoint
        next_adjoint = torch.autograd.grad(local_lagrangian, state_t)[0]

    contributions = []
    parameters = tuple(model.parameters())
    for time_index in range(a, b + 1):
        # State and adjoint are fixed in D_theta; only the occurrence at t is active.
        next_state, prediction = model(states[time_index], u[time_index])
        local_lagrangian = _stage_loss(prediction, y[time_index]) + torch.dot(
            next_state, adjoint_after[time_index - a])
        gradient = torch.autograd.grad(local_lagrangian, parameters, allow_unused=True)
        contributions.append(torch.cat([
            (torch.zeros_like(parameter) if value is None else value).reshape(-1)
            for parameter, value in zip(parameters, gradient)
        ]))
    return contributions
