"""Compute Figure 2 pointwise errors against the full training-horizon reference."""

import argparse
import csv
import json
from pathlib import Path

import torch

from gradient_turnpike.data import load_wh
from gradient_turnpike.evaluation import load_model
from gradient_turnpike.gradients import pointwise_gradient_contributions
from gradient_turnpike.training import prepare_sequences
from gradient_turnpike.utils import (
    OUTPUT_ROOT, load_config, prepare_output_directory, resolve_device, runtime_metadata,
)


DEFAULT_CONFIG = Path("configs/wh_paper.json")


def save_csv(path: Path, rows: list[dict]) -> None:
    """Save a nonempty pointwise analysis table with project-relative provenance."""
    if not rows:
        raise ValueError("Cannot save an empty pointwise analysis table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nonnegative_integer_list(text: str) -> list[int]:
    """Parse comma-separated nonnegative symmetric context extensions."""
    try:
        values = sorted({int(value) for value in text.split(",")})
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use comma-separated nonnegative integers") from error
    if not values or values[0] < 0:
        raise argparse.ArgumentTypeError("context extensions must be nonnegative")
    return values


def pointwise_rows(full: list[torch.Tensor], truncated: list[torch.Tensor], *,
                   update_start: int, T_P: int, T_F: int) -> list[dict]:
    """Compare matching adjoint-form contributions without curve-wise scaling."""
    epsilon = torch.finfo(full[0].dtype).eps
    rows = []
    for relative_t, (full_value, truncated_value) in enumerate(zip(full, truncated)):
        full_norm = torch.linalg.vector_norm(full_value).item()
        truncated_norm = torch.linalg.vector_norm(truncated_value).item()
        error = torch.linalg.vector_norm(full_value - truncated_value).item()
        rows.append({"t": update_start + relative_t, "relative_t": relative_t,
                     "T_P": T_P, "T_F": T_F,
                     "full_gradient_norm": full_norm,
                     "truncated_gradient_norm": truncated_norm,
                     "gradient_error": error,
                     "relative_gradient_error": error / max(full_norm, epsilon)})
    return rows


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_path, _ = config_parser.parse_known_args()
    base_config = load_config(config_path.config)
    config = base_config["gradient_turnpike"]
    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--checkpoint", type=Path, default=Path(config["checkpoint"]))
    parser.add_argument("--update-start", type=int, default=config["update_start"])
    parser.add_argument("--update-length", type=int, default=config["update_length"])
    parser.add_argument("--symmetric-contexts", dest="symmetric_contexts",
                        type=nonnegative_integer_list, default=config["symmetric_contexts"],
                        help="Comma-separated symmetric T_P=T_F values.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.update_length < 2:
        parser.error("--update-length must be at least 2")
    if not args.symmetric_contexts or min(args.symmetric_contexts) != 0:
        parser.error("--symmetric-contexts must be nonnegative and include 0")
    try:
        device = resolve_device(args.device)
    except RuntimeError as error:
        parser.error(str(error))
    if not args.checkpoint.exists():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    output_dir = OUTPUT_ROOT / "analysis" / "gradient_turnpike" / "pointwise"
    try:
        prepare_output_directory(output_dir, overwrite=args.overwrite)
    except FileExistsError as error:
        parser.error(str(error))
    data = load_wh(base_config["data"]["train_fraction"])
    model, checkpoint_metadata = load_model(args.checkpoint, device=device, dtype=torch.float64)
    (u, y), = prepare_sequences(model, data.train)
    a_global = args.update_start
    b_global = a_global + args.update_length - 1
    margin = max(args.symmetric_contexts)
    if a_global < margin or b_global + margin >= len(u):
        parser.error("The update interval and all truncated contexts must fit in the training trajectory")
    print(f"Pointwise gradient analysis: U=[{a_global}, {b_global}], "
          f"full context=[0, {len(u) - 1}], device={device}, dtype=float64", flush=True)
    full = pointwise_gradient_contributions(model, u, y, a=a_global, b=b_global)
    full_gradient = torch.stack(full).mean(dim=0)

    rows_by_context: dict[tuple[int, int], list[dict]] = {}
    for context in args.symmetric_contexts:
        start = a_global - context
        stop = b_global + context + 1
        truncated = pointwise_gradient_contributions(
            model, u[start:stop], y[start:stop],
            a=context, b=context + args.update_length - 1)
        rows_by_context[context, context] = pointwise_rows(
            full, truncated, update_start=a_global, T_P=context, T_F=context)

    rows = [row for context_rows in rows_by_context.values() for row in context_rows]
    save_csv(output_dir / "pointwise_gradient_errors.csv", rows)
    metadata = {
        "checkpoint": str(args.checkpoint), "configuration_path": str(args.config),
        "configured_gradient_turnpike": config,
        "data_split": "WH training trajectory only", "dtype": "float64",
        "update_interval": [a_global, b_global], "update_length": args.update_length,
        "full_context": [0, len(u) - 1],
        "full_context_T_P": a_global, "full_context_T_F": len(u) - 1 - b_global,
        "symmetric_contexts": args.symmetric_contexts,
        "full_gradient_norm": torch.linalg.vector_norm(full_gradient).item(),
        "relative_error_denominator": "max(norm(g_t), torch.finfo(float64).eps) at each t",
        "train_fraction": base_config["data"]["train_fraction"],
        "seed": checkpoint_metadata.get("seed"), "device": device,
        **runtime_metadata(device),
    }
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved Figure-2 analysis data to {output_dir}")


if __name__ == "__main__":
    main()
