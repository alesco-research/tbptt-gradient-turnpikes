"""Wiener–Hammerstein data in chronological, unnormalized CPU tensors."""

from dataclasses import dataclass

import nonlinear_benchmarks
import torch
from nonlinear_benchmarks.utilities import Input_output_data


@dataclass(frozen=True)
class SignalData:
    """Signals shaped (time, channels), with sampling time in seconds."""

    u: torch.Tensor
    y: torch.Tensor
    sampling_time: float
    state_initialization_window_length: int = 0


@dataclass(frozen=True)
class WHData:
    train: SignalData
    validation: SignalData
    test: SignalData
    official_test: Input_output_data


def split_train_validation(
    data: SignalData, train_fraction: float = 0.8
) -> tuple[SignalData, SignalData]:
    """Split at floor(N * train_fraction), retaining order and every sample.

    Only pass the development sequence here, never the official test set.
    Returned tensors own their storage so in-place preprocessing stays local.
    No initialization window is assigned to these development subsets.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    if data.u.ndim != 2 or data.y.ndim != 2:
        raise ValueError("u and y must have shape (time, channels)")
    if len(data.u) != len(data.y):
        raise ValueError("u and y must have the same number of samples")
    split = int(len(data.u) * train_fraction)
    if not 0 < split < len(data.u):
        raise ValueError("Both training and validation subsets must be nonempty")
    return tuple(
        SignalData(data.u[s].clone(), data.y[s].clone(), data.sampling_time)
        for s in (slice(None, split), slice(split, None))
    )


def load_wh(train_fraction: float = 0.8) -> WHData:
    """Load the official benchmark and split only its development sequence.

    The upstream package downloads/caches the benchmark on first use. The
    official test sequence includes its entire initialization window; nothing
    is shuffled, normalized, or removed. All signals become CPU float32 copies.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    train_val, test = nonlinear_benchmarks.WienerHammerBenchMark(atleast_2d=True)

    def convert(sequence, initialization_window=0):
        return SignalData(
            u=torch.tensor(sequence.u, dtype=torch.float32, device="cpu"),
            y=torch.tensor(sequence.y, dtype=torch.float32, device="cpu"),
            sampling_time=float(sequence.sampling_time),
            state_initialization_window_length=initialization_window,
        )

    train, validation = split_train_validation(convert(train_val), train_fraction)
    return WHData(
        train, validation,
        convert(test, int(test.state_initialization_window_length)),
        test,
    )
