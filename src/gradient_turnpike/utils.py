"""Small, project-relative utilities for reproducible experiments."""

from __future__ import annotations

from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
from typing import Any

import torch


OUTPUT_ROOT = Path("outputs")
FIGURE_ROOT = Path("figures")


def load_config(path: Path) -> dict[str, Any]:
    """Read one JSON experiment configuration."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` while requiring CUDA when it is selected explicitly."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def set_reproducibility(seed: int) -> None:
    """Set deterministic PyTorch and Python random state before model creation."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def runtime_metadata(device: str) -> dict[str, Any]:
    """Return portable environment information recorded with each run."""
    return {
        "versions": {name: version(name) for name in ("torch", "nonlinear-benchmarks", "matplotlib")},
        "python_version": platform.python_version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name() if device == "cuda" else "cpu",
    }


def bptt_run_directory(run_id: str = "001") -> Path:
    return OUTPUT_ROOT / "bptt" / f"run_{run_id}"


def tbptt_run_directory(T_P: int, T_F: int, run_id: str = "001") -> Path:
    return OUTPUT_ROOT / "tbptt" / f"TP_{T_P}_TF_{T_F}" / f"run_{run_id}"


def prepare_output_directory(path: Path, *, overwrite: bool = False) -> Path:
    """Create an output directory without silently overwriting artifacts."""
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} already contains results; choose another run ID or use --overwrite")
    path.mkdir(parents=True, exist_ok=True)
    return path
