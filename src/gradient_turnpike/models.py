"""Vanilla recurrent state-space model with the manuscript's time indexing."""

import torch
from torch import nn


class VanillaRNN(nn.Module):
    """Single-layer tanh RNN: x[t+1] = f(x[t], u[t]), y[t] = h(x[t], u[t]).

    The readout is affine in the current state and input. PyTorch initializes
    the parameters; set torch.manual_seed before constructing the model.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 16, output_size: int = 1):
        super().__init__()
        if min(input_size, hidden_size, output_size) < 1:
            raise ValueError("Model dimensions must be positive")
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity="tanh")
        self.readout = nn.Linear(hidden_size + input_size, output_size)

    def f(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Advance one state; x and u are one-dimensional vectors."""
        _, next_state = self.rnn(u.reshape(1, 1, -1), x.reshape(1, 1, -1))
        return next_state[0, 0]

    def h(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Read out current states/inputs, with channels on the last axis."""
        return self.readout(torch.cat((x, u), dim=-1))

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x_next, y) for one time step."""
        return self.f(x, u), self.h(x, u)
