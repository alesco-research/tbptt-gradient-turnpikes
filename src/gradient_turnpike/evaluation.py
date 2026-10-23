"""Free-running WH evaluation and concise output-fit diagnostics."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from nonlinear_benchmarks.error_metrics import RMSE
from nonlinear_benchmarks.utilities import Input_output_data
from torch import nn

from gradient_turnpike.data import SignalData, WHData
from gradient_turnpike.models import VanillaRNN
from gradient_turnpike.simulation import simulate
from gradient_turnpike.training import prepare_sequences


def load_model(checkpoint: Path, *, device: str, dtype: torch.dtype) -> tuple[VanillaRNN, dict]:
    """Load a portable project checkpoint in eval mode."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    model = VanillaRNN(**metadata["model_config"]).to(device=device, dtype=dtype)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, metadata


@torch.no_grad()
def simulate_validation(model: VanillaRNN, train: SignalData,
                        validation: SignalData) -> tuple[torch.Tensor, torch.Tensor]:
    """Free-run validation after replaying the contiguous training inputs."""
    (u_train, _), (u_val, y_val) = prepare_sequences(model, train, validation)
    _, states = simulate(model, u_train)
    prediction, _ = simulate(model, u_val, states[-1])
    return prediction, y_val


def official_wh_rmse(y_true: np.ndarray | Input_output_data, prediction: torch.Tensor) -> float:
    """Return the scalar WH benchmark RMSE for an already simulated output."""
    value = np.asarray(RMSE(y_true, prediction.detach().cpu().numpy()))
    if value.size != 1:
        raise ValueError("the Wiener--Hammerstein benchmark must have one output channel")
    return float(value.squeeze())


@torch.no_grad()
def evaluate_test(model: VanillaRNN, test: SignalData) -> float:
    """Free-run official test data and score after its initialization window."""
    u_test, y_test = prepare_sequences(model, test)[0]
    prediction, _ = simulate(model, u_test)
    warmup = test.state_initialization_window_length
    return nn.functional.mse_loss(prediction[warmup:], y_test[warmup:]).item()


@torch.no_grad()
def evaluate_wh_test_metrics(model: VanillaRNN, test: SignalData,
                             official_test: Input_output_data) -> tuple[float, float]:
    """Return repository test MSE and official WH test RMSE from one simulation."""
    u_test, y_test = prepare_sequences(model, test)[0]
    prediction, _ = simulate(model, u_test)
    warmup = test.state_initialization_window_length
    mse = nn.functional.mse_loss(prediction[warmup:], y_test[warmup:]).item()
    return mse, official_wh_rmse(official_test, prediction)


@torch.no_grad()
def evaluate_wh_training_metrics(model: VanillaRNN, train: SignalData) -> tuple[float, float]:
    """Return repository training MSE and official WH RMSE from one simulation."""
    u_train, y_train = prepare_sequences(model, train)[0]
    model.eval()
    prediction, _ = simulate(model, u_train)
    mse = nn.functional.mse_loss(prediction, y_train).item()
    return mse, official_wh_rmse(y_train.detach().cpu().numpy(), prediction)


def save_validation_diagnostics(output_dir: Path, model: VanillaRNN, data: WHData,
                                *, segment_length: int = 2000) -> float:
    """Save free-running output and residual figures, returning validation MSE."""
    if segment_length < 1:
        raise ValueError("segment_length must be positive")
    prediction, target = simulate_validation(model, data.train, data.validation)
    mse = nn.functional.mse_loss(prediction, target).item()
    count = min(segment_length, len(target))
    start = max(0, (len(target) - count) // 2)
    stop = start + count
    time = torch.arange(start, stop, dtype=torch.float64) * data.validation.sampling_time
    measured = target[start:stop].detach().cpu()
    simulated = prediction[start:stop].detach().cpu()
    residual = measured - simulated

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(time, measured, label=r"Measured $y_t$", linewidth=1.2)
    ax.plot(time, simulated, label=r"Simulated $\hat y_t$", linewidth=1.0)
    ax.set(xlabel="Time within validation sequence [s]", ylabel="Output")
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"validation_output_fit.{suffix}", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(time, residual, linewidth=1.0, color="C3")
    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set(xlabel="Time within validation sequence [s]", ylabel=r"Residual $y_t-\hat y_t$")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"validation_residual.{suffix}", dpi=160)
    plt.close(fig)
    return mse


def save_evaluation(output_dir: Path, model: VanillaRNN, data: WHData,
                    *, segment_length: int = 2000, include_test: bool = False) -> dict:
    """Persist validation diagnostics; include official test only when requested."""
    results = {"validation_mse": save_validation_diagnostics(
        output_dir, model, data, segment_length=segment_length)}
    if include_test:
        results["test_mse"] = evaluate_test(model, data.test)
    (output_dir / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
    return results
