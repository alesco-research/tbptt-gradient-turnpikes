"""Independent float64 references for a small tanh state-space model."""

import torch


def make_parameters(state_dim=3, input_dim=1, output_dim=1, seed=7):
    generator = torch.Generator().manual_seed(seed)
    A = 0.15 * torch.randn(state_dim, state_dim, generator=generator, dtype=torch.float64)
    B = 0.20 * torch.randn(state_dim, input_dim, generator=generator, dtype=torch.float64)
    b = 0.05 * torch.randn(state_dim, generator=generator, dtype=torch.float64)
    C = 0.30 * torch.randn(output_dim, state_dim, generator=generator, dtype=torch.float64)
    D = 0.10 * torch.randn(output_dim, input_dim, generator=generator, dtype=torch.float64)
    d = 0.05 * torch.randn(output_dim, generator=generator, dtype=torch.float64)
    return {name: torch.nn.Parameter(value)
            for name, value in {"A": A, "B": B, "b": b, "C": C, "D": D, "d": d}.items()}


def dynamics(x, u, parameters):
    return torch.tanh(parameters["A"] @ x + parameters["B"] @ u + parameters["b"])


def output(x, u, parameters):
    return parameters["C"] @ x + parameters["D"] @ u + parameters["d"]


def detach_parameters(parameters):
    """Freeze parameter values while preserving state derivatives."""
    return {name: parameter.detach() for name, parameter in parameters.items()}


def gradients_from_loss(loss, parameters):
    gradients = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True)
    return {name: gradient if gradient is not None else torch.zeros_like(parameter)
            for (name, parameter), gradient in zip(parameters.items(), gradients)}


def restricted_context_gradient(parameters, u, y, x_r, a, b, r, s):
    """Average occurrences in U=[a,b] using C=[r,s], initialized at x_r.

    Future losses contribute through the state, but direct outside-U parameter
    occurrences are frozen. Normalize by |U|, not by the context length.
    """
    assert 0 <= r <= a <= b <= s < len(u)
    frozen = detach_parameters(parameters)
    x = x_r
    loss = x.new_zeros(())
    for t in range(r, s + 1):
        active = parameters if a <= t <= b else frozen
        prediction = output(x, u[t], active)
        if t >= a:  # Past losses have no derivative in the active occurrences.
            loss = loss + 0.5 * (prediction - y[t]).square().sum()
        x = dynamics(x, u[t], active)  # Keep the future state connection.
    return gradients_from_loss(loss / (b - a + 1), parameters)


def one_sided_gradient(parameters, u, y, x_r, a, b, r):
    """Independent past burn-in followed by ordinary AD on U, for T_F=0."""
    frozen = detach_parameters(parameters)
    x = x_r
    for t in range(r, a):
        x = dynamics(x, u[t], frozen)
    loss = x.new_zeros(())
    for t in range(a, b + 1):
        prediction = output(x, u[t], parameters)
        loss = loss + 0.5 * (prediction - y[t]).square().sum()
        x = dynamics(x, u[t], parameters)
    return gradients_from_loss(loss / (b - a + 1), parameters)


def full_bptt_gradient(parameters, u, y, x0):
    x = x0
    loss = x.new_zeros(())
    for t in range(len(u)):
        prediction = output(x, u[t], parameters)
        loss = loss + 0.5 * (prediction - y[t]).square().sum()
        x = dynamics(x, u[t], parameters)
    return gradients_from_loss(loss / len(u), parameters)


def propagate_state_without_parameter_gradient(parameters, u, x0, steps):
    x = x0
    with torch.no_grad():
        for t in range(steps):
            x = dynamics(x, u[t], parameters)
    return x


def make_test_problem():
    generator = torch.Generator().manual_seed(123)
    u = torch.randn(12, 1, generator=generator, dtype=torch.float64)
    y = torch.randn(12, 1, generator=generator, dtype=torch.float64)
    x0 = 0.1 * torch.randn(3, generator=generator, dtype=torch.float64)
    return make_parameters(), u, y, x0


def test_tf_zero_equals_one_sided_tbptt():
    parameters, u, y, x0 = make_test_problem()
    a, b, r = 4, 7, 1
    x_r = propagate_state_without_parameter_gradient(parameters, u, x0, r)
    generic = restricted_context_gradient(parameters, u, y, x_r, a, b, r, s=b)
    one_sided = one_sided_gradient(parameters, u, y, x_r, a, b, r)
    for name in parameters:
        torch.testing.assert_close(generic[name], one_sided[name], rtol=1e-10, atol=1e-12)


def test_future_context_changes_dynamic_parameter_gradient():
    parameters, u, y, x0 = make_test_problem()
    a, b, r = 4, 7, 1
    x_r = propagate_state_without_parameter_gradient(parameters, u, x0, r)
    no_future = restricted_context_gradient(parameters, u, y, x_r, a, b, r, s=b)
    future = restricted_context_gradient(parameters, u, y, x_r, a, b, r, s=b + 3)
    for name in ("A", "B"):
        assert torch.linalg.vector_norm(future[name] - no_future[name]) > 1e-10


def test_future_context_has_no_direct_output_parameter_contribution():
    parameters, u, y, x0 = make_test_problem()
    a, b, r = 4, 7, 1
    x_r = propagate_state_without_parameter_gradient(parameters, u, x0, r)
    no_future = restricted_context_gradient(parameters, u, y, x_r, a, b, r, s=b)
    future = restricted_context_gradient(parameters, u, y, x_r, a, b, r, s=b + 3)
    for name in ("C", "D", "d"):
        torch.testing.assert_close(future[name], no_future[name], rtol=1e-10, atol=1e-12)


def test_complete_context_equals_full_bptt():
    parameters, u, y, x0 = make_test_problem()
    full = full_bptt_gradient(parameters, u, y, x0)
    truncated = restricted_context_gradient(parameters, u, y, x0, a=0, b=len(u) - 1, r=0, s=len(u) - 1)
    for name in parameters:
        torch.testing.assert_close(truncated[name], full[name], rtol=1e-10, atol=1e-12)


def test_gradients_are_finite_and_have_correct_shape():
    parameters, u, y, x0 = make_test_problem()
    a, b, r, s = 4, 7, 1, 9
    x_r = propagate_state_without_parameter_gradient(parameters, u, x0, r)
    gradients = restricted_context_gradient(parameters, u, y, x_r, a, b, r, s)
    for name, parameter in parameters.items():
        assert gradients[name].shape == parameter.shape
        assert torch.isfinite(gradients[name]).all()
