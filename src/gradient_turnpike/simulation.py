"""Differentiable open-loop simulation; no target feedback or detachment."""

import torch

from gradient_turnpike.models import VanillaRNN


# cudnnSetRNNDataDescriptor rejects maxSeqLength > 65535.
# https://docs.nvidia.com/deeplearning/cudnn/backend/v9.17.1/api/cudnn-adv-library.html#cudnnsetrnndatadescriptor
CUDNN_MAX_SEQUENCE_LENGTH = 65_535


def simulate(
    model: VanillaRNN, u: torch.Tensor, x0: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predictions (N, output_size) and states (N+1, hidden_size).

    Inputs have shape (N, input_size). The initial state defaults to zero.
    Inputs and x0 must match the model's device and dtype. The entire graph is
    retained, including gradients with respect to a supplied initial state.
    Use torch.no_grad() explicitly when gradients are not needed.

    CUDA sequences exceeding cuDNN's length limit use multiple RNN calls,
    carrying the hidden state without detaching it. Gradients still span the
    complete sequence; these call boundaries do not truncate BPTT.
    """
    if u.ndim != 2 or len(u) == 0 or u.shape[1] != model.rnn.input_size:
        raise ValueError("u must have nonempty shape (time, input_size)")
    if x0 is None:
        x0 = u.new_zeros(model.rnn.hidden_size)
    if x0.shape != (model.rnn.hidden_size,):
        raise ValueError("x0 must have shape (hidden_size,)")
    # One sequence, batch size one. RNN outputs are x[1], ..., x[N].
    call_length = CUDNN_MAX_SEQUENCE_LENGTH if u.is_cuda else len(u)
    hidden = x0.reshape(1, 1, -1).contiguous()
    state_parts = [x0.unsqueeze(0)]
    for inputs in u.split(call_length):
        next_states, hidden = model.rnn(inputs.unsqueeze(1).contiguous(), hidden)
        state_parts.append(next_states[:, 0])
    states = torch.cat(state_parts, dim=0)
    y = model.h(states[:-1], u)
    return y, states
