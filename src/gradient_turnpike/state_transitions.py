"""State Jacobians and finite-step sensitivities for the tanh RNN."""

from dataclasses import dataclass

import torch

from gradient_turnpike.models import VanillaRNN


@dataclass(frozen=True)
class UniformStateSearchResult:
    """Best value found by a numerical search over starts and box states."""

    worst_value_found: float
    worst_start_index: int
    worst_initial_state: list[float]
    settings: dict[str, object]


def rnn_dynamics_parameters(model: VanillaRNN) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return A, B, b, summing PyTorch's two trainable recurrent biases."""
    rnn = model.rnn
    return rnn.weight_hh_l0, rnn.weight_ih_l0, rnn.bias_ih_l0 + rnn.bias_hh_l0


def admissible_start_indices(transition_count: int, horizon: int) -> torch.Tensor:
    """Return exactly the starts ``0..N-T`` for length-``T`` input slices."""
    if horizon < 1 or horizon > transition_count:
        raise ValueError("horizon must be between one and transition_count")
    return torch.arange(transition_count - horizon + 1, dtype=torch.long)


def transition_product_norms(
    recurrent_matrix: torch.Tensor, input_matrix: torch.Tensor, bias: torch.Tensor,
    inputs: torch.Tensor, starts: torch.Tensor, initial_states: torch.Tensor, horizon: int,
) -> torch.Tensor:
    """Float64 product 2-norms for paired input starts and arbitrary box states."""
    if recurrent_matrix.dtype != torch.float64 or initial_states.dtype != torch.float64:
        raise ValueError("state-transition analysis requires float64")
    x = initial_states
    product = torch.eye(recurrent_matrix.shape[0], dtype=torch.float64, device=x.device).expand(len(x), -1, -1).clone()
    for offset in range(horizon):
        u = inputs.index_select(0, starts + offset)
        x = torch.tanh(x @ recurrent_matrix.T + u @ input_matrix.T + bias)
        jacobian = (1.0 - x.square()).unsqueeze(-1) * recurrent_matrix
        product = torch.bmm(jacobian, product)
    return torch.linalg.svdvals(product)[..., 0]


def projected_spectral_ascent(
    recurrent_matrix: torch.Tensor, input_matrix: torch.Tensor, bias: torch.Tensor,
    inputs: torch.Tensor, starts: torch.Tensor, initial_states: torch.Tensor, horizon: int,
    *, iterations: int, learning_rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retain each candidate's best projected-ascent iterate in ``[-1,1]^n``."""
    x = initial_states.detach().clone().requires_grad_(True)
    with torch.no_grad():
        best_values = transition_product_norms(recurrent_matrix, input_matrix, bias, inputs, starts, x, horizon)
        best_states = x.detach().clone()
    optimizer = torch.optim.Adam([x], lr=learning_rate)
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        values = transition_product_norms(recurrent_matrix, input_matrix, bias, inputs, starts, x, horizon)
        (-values.sum()).backward()
        with torch.no_grad():
            improved = values.detach() > best_values
            best_values[improved] = values.detach()[improved]
            best_states[improved] = x.detach()[improved]
        optimizer.step()
        with torch.no_grad():
            x.clamp_(-1.0, 1.0)
    with torch.no_grad():
        values = transition_product_norms(recurrent_matrix, input_matrix, bias, inputs, starts, x, horizon)
        improved = values > best_values
        best_values[improved] = values[improved]
        best_states[improved] = x.detach()[improved]
    return best_values, best_states


def paper_search_settings(seed: int, *, smoke: bool = False) -> dict[str, object]:
    """Resolve the audited deterministic paper search settings for one seed."""
    if seed not in (1, 2, 3, 4, 5):
        raise ValueError("paper search is defined for seeds 1--5")
    broad = seed == 3
    settings = {
        "horizon": 20, "screen_restarts": 33, "screen_seed": 2701,
        "screen_start_stride": 1,
        "candidate_windows": 1000 if broad else 300,
        "sobol": 8 if broad else 4, "random": 8 if broad else 4,
        "corners": 8 if broad else 4, "axis_boundary": 8,
        "first_iterations": 75 if broad else 50,
        "final_windows": 150 if broad else 50,
        "final_iterations": 250 if broad else 125,
        "learning_rates": (0.01, 0.03, 0.06),
    }
    if smoke:
        settings.update(screen_start_stride=1024, candidate_windows=4, screen_restarts=3, sobol=1, random=1, corners=1,
                        axis_boundary=2, first_iterations=2, final_windows=2, final_iterations=3,
                        learning_rates=(0.03,))
    return settings


def deterministic_restart_family(dimension: int, *, sobol: int, random: int, corners: int,
                                 axis_boundary: int, seed: int) -> torch.Tensor:
    """Deterministic zero/interior/corner/axis-boundary states in the unit box."""
    generator = torch.Generator().manual_seed(seed)
    quasi = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed)
    parts = [torch.zeros((1, dimension), dtype=torch.float64),
             2 * quasi.draw(sobol).to(torch.float64) - 1,
             2 * torch.rand((random, dimension), generator=generator, dtype=torch.float64) - 1,
             2 * torch.randint(0, 2, (corners, dimension), generator=generator, dtype=torch.float64) - 1]
    axes = torch.eye(dimension, dtype=torch.float64)
    parts.append(torch.cat((axes, -axes))[:axis_boundary])
    return torch.cat(parts)


def screen_uniform_state_windows(
    recurrent_matrix: torch.Tensor, input_matrix: torch.Tensor, bias: torch.Tensor,
    inputs: torch.Tensor, screening_states: torch.Tensor, *, horizon: int,
    candidate_windows: int, start_stride: int = 1, chunk_size: int = 1024,
) -> list[tuple[float, int, torch.Tensor]]:
    """Screen SVD product norms at all starts (or a strided smoke subset)."""
    if start_stride < 1:
        raise ValueError("start_stride must be positive")
    starts_all = admissible_start_indices(len(inputs), horizon)[::start_stride]
    candidates: list[tuple[float, int, torch.Tensor]] = []
    for lower in range(0, len(starts_all), chunk_size):
        starts = starts_all[lower:lower + chunk_size]
        paired_starts = starts.repeat_interleave(len(screening_states))
        paired_states = screening_states.repeat(len(starts), 1)
        with torch.no_grad():
            values = transition_product_norms(
                recurrent_matrix, input_matrix, bias, inputs, paired_starts,
                paired_states, horizon,
            ).reshape(len(starts), len(screening_states))
        best_values, best_indices = values.max(dim=1)
        candidates.extend((float(value), int(start), screening_states[int(index)])
                          for value, start, index in zip(best_values, starts, best_indices))
    return sorted(candidates, key=lambda item: item[0], reverse=True)[:candidate_windows]


def search_uniform_state_contraction(
    model: VanillaRNN, inputs: torch.Tensor, *, seed: int, smoke: bool = False,
) -> UniformStateSearchResult:
    """Search box states and input starts; this is not a certified supremum."""
    settings = paper_search_settings(seed, smoke=smoke)
    horizon = int(settings["horizon"])
    recurrent_matrix, input_matrix, bias = rnn_dynamics_parameters(model)
    dimension = recurrent_matrix.shape[0]
    screening_states = deterministic_restart_family(
        dimension, sobol=int(settings["screen_restarts"]) - 1, random=0,
        corners=0, axis_boundary=0, seed=int(settings["screen_seed"]),
    )
    candidates = screen_uniform_state_windows(
        recurrent_matrix, input_matrix, bias, inputs, screening_states,
        horizon=horizon, candidate_windows=int(settings["candidate_windows"]),
        start_stride=int(settings["screen_start_stride"]),
    )
    restart_family = deterministic_restart_family(
        dimension, sobol=int(settings["sobol"]), random=int(settings["random"]),
        corners=int(settings["corners"]), axis_boundary=int(settings["axis_boundary"]),
        seed=9041 + seed,
    )
    starts: list[int] = []
    populations = []
    for _, start, screening_state in candidates:
        population = torch.cat((screening_state[None], restart_family))
        starts.extend([start] * len(population))
        populations.append(population)
    paired_starts = torch.tensor(starts, dtype=torch.long)
    initial_states = torch.cat(populations)
    values, states = projected_spectral_ascent(
        recurrent_matrix, input_matrix, bias, inputs, paired_starts, initial_states,
        horizon, iterations=int(settings["first_iterations"]), learning_rate=.03,
    )
    strongest = torch.topk(values, min(int(settings["final_windows"]), len(values))).indices
    best_index = int(values.argmax())
    best = (float(values[best_index]), int(paired_starts[best_index]), states[best_index].tolist())
    for learning_rate in settings["learning_rates"]:
        refined_values, refined_states = projected_spectral_ascent(
            recurrent_matrix, input_matrix, bias, inputs, paired_starts[strongest],
            states[strongest], horizon, iterations=int(settings["final_iterations"]),
            learning_rate=float(learning_rate),
        )
        index = int(refined_values.argmax())
        candidate = (float(refined_values[index]), int(paired_starts[strongest][index]),
                     refined_states[index].tolist())
        if candidate[0] > best[0]:
            best = candidate
    return UniformStateSearchResult(*best, settings=settings)


def analytic_state_jacobians(states: torch.Tensor, recurrent_matrix: torch.Tensor) -> torch.Tensor:
    """Return J[t] = diag(1-states[t+1]**2) @ A; states[0] is the initial state."""
    if states.ndim != 2 or len(states) < 2:
        raise ValueError("states must have shape (time + 1, state_dim) with time >= 1")
    if recurrent_matrix.ndim != 2 or recurrent_matrix.shape[0] != recurrent_matrix.shape[1]:
        raise ValueError("recurrent_matrix must be square")
    if states.shape[1] != recurrent_matrix.shape[0]:
        raise ValueError("state and recurrent-matrix dimensions must agree")
    derivative = 1.0 - states[1:].square()
    return derivative.unsqueeze(-1) * recurrent_matrix.unsqueeze(0)
