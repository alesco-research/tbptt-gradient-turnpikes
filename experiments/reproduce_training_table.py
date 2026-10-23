"""Print Table I from saved per-seed metrics; optionally rescore its checkpoints."""

import argparse
import csv
from pathlib import Path
from statistics import median

from gradient_turnpike.metrics import paired_ratio_median


RESULTS = Path("outputs/analysis/training_comparison/per_run_metrics.csv")


def checkpoint_path(row: dict) -> Path:
    seed = int(row["seed"])
    if row["method"] == "BPTT":
        return Path(f"outputs/bptt/training_comparison_final/bptt_seed_{seed:03d}/best_model.pt")
    context = int(row["T_P"])
    return Path(f"outputs/tbptt/training_comparison_final/tbptt_T{context:02d}_seed_{seed:03d}/best_model.pt")


def verify_checkpoints(rows: list[dict], device: str) -> None:
    """Compare free-running train/validation/test scores with the archived rows."""
    import math
    import torch

    from gradient_turnpike.data import load_wh
    from gradient_turnpike.evaluation import (
        evaluate_wh_test_metrics, evaluate_wh_training_metrics, load_model,
    )
    from gradient_turnpike.training import evaluate_validation, prepare_sequences

    data = load_wh()
    maximum = 0.0
    for row in rows:
        path = checkpoint_path(row)
        model, metadata = load_model(path, device=device, dtype=torch.float32)
        if (int(metadata["seed"]) != int(row["seed"]) or
                int(metadata["checkpoint_update"]) != int(row["best_validation_update"])):
            raise ValueError(f"Checkpoint metadata disagree with saved metrics: {path}")
        (u_train, _), (u_val, y_val) = prepare_sequences(model, data.train, data.validation)
        train_mse, train_rmse = evaluate_wh_training_metrics(model, data.train)
        test_mse, test_rmse = evaluate_wh_test_metrics(model, data.test, data.official_test)
        actual = {"best_validation_mse": evaluate_validation(model, u_train, u_val, y_val),
                  "full_training_mse": train_mse, "full_training_rmse": train_rmse,
                  "test_mse": test_mse, "test_rmse": test_rmse}
        for name, value in actual.items():
            expected = float(row[name])
            maximum = max(maximum, abs(value - expected))
            if not math.isclose(value, expected, rel_tol=5e-5, abs_tol=5e-7):
                raise ValueError(f"{path}: {name}={value}, archived value={expected}")
    print(f"Verified {len(rows)} checkpoints; maximum absolute score difference: {maximum:.3g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true", help="Rescore all 25 selected models; requires data.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    with RESULTS.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    references = {int(row["seed"]): row for row in rows if row["method"] == "BPTT"}
    if len(rows) != 25 or set(references) != set(range(1, 6)):
        raise ValueError("Expected the five-seed, 25-model comparison")
    print("| Method | T_P=T_F | Train RMSE | Test RMSE | Train/BPTT | Test/BPTT | Speedup |")
    print("|---|---|---|---|---|---|---|")
    for method, context in [("BPTT", "full"), *(("TBPTT", str(c)) for c in (0, 10, 20, 40))]:
        group = {int(row["seed"]): row for row in rows if row["method"] == method and row["T_P"] == context}
        if len(group) != 5:
            raise ValueError(f"Expected five distinct seeds for {method}, context={context}")
        med = lambda name: median(float(row[name]) for row in group.values())
        ratios = [paired_ratio_median({s: float(r[name]) for s, r in group.items()},
                                     {s: float(r[name]) for s, r in references.items()})
                  for name in ("full_training_rmse", "test_rmse")]
        speedup = median(float(references[s]["training_time_seconds"]) / float(r["training_time_seconds"])
                         for s, r in group.items())
        print(f"| {method} | {context} | {med('full_training_rmse'):.5f} | {med('test_rmse'):.5f} | "
              f"{ratios[0]:.2f} | {ratios[1]:.2f} | {speedup:.2f} |")
    if args.evaluate:
        from gradient_turnpike.utils import resolve_device
        verify_checkpoints(rows, resolve_device(args.device))


if __name__ == "__main__":
    main()
