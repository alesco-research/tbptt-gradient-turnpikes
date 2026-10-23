"""Search 20-step Jacobian norms on WH inputs and states in [-1,1]^16.

--paper uses the saved search settings; --smoke reduces the search counts.
Neither certifies the continuous-domain supremum.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from gradient_turnpike.data import load_wh
from gradient_turnpike.evaluation import load_model
from gradient_turnpike.state_transitions import search_uniform_state_contraction
from gradient_turnpike.training import prepare_sequences
from gradient_turnpike.utils import load_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/wh_paper.json"
ANALYSIS_DIRECTORY = ROOT / "outputs" / "analysis" / "uniform_state_contraction"


def final_bptt_checkpoint(seed: int) -> Path:
    """Return the fixed-parameter BPTT checkpoint used by the paper search."""
    return ROOT / "outputs" / "bptt" / f"run_seed_{seed:03d}" / "final_model.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Base WH configuration; supplies the development split and state dimension.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--paper", action="store_true", help="Run the complete deterministic paper search.")
    mode.add_argument("--smoke", action="store_true", help="Run a reduced deterministic search for one or all seeds.")
    parser.add_argument("--seed", type=int, help="Optional one-based BPTT seed; useful with --smoke.")
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIRECTORY,
                        help="Directory for untracked recomputation rows; never overwrites frozen compact summaries.")
    args = parser.parse_args()
    base = load_config(args.config)
    data = load_wh(base["data"]["train_fraction"])
    seeds = (args.seed,) if args.seed is not None else tuple(base["reproducibility"]["seeds"])
    rows = []
    for seed in seeds:
        checkpoint = final_bptt_checkpoint(seed)
        model, metadata = load_model(checkpoint, device="cpu", dtype=torch.float64)
        if metadata.get("state_dimension", model.rnn.hidden_size) != base["model"]["state_dim"]:
            raise ValueError(f"{checkpoint} has an incompatible state dimension")
        (u_train, _), (u_validation, _) = prepare_sequences(model, data.train, data.validation)
        development_inputs = torch.cat((u_train, u_validation))
        result = search_uniform_state_contraction(model, development_inputs, seed=seed, smoke=args.smoke)
        rows.append({
            "seed": seed,
            "horizon": int(result.settings["horizon"]),
            "worst_value_found": result.worst_value_found,
            "worst_start_index": result.worst_start_index,
            "numerical_counterexample_found": result.worst_value_found > 1.0,
            "search_settings": json.dumps(result.settings, sort_keys=True),
        })
        print(rows[-1], flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = "recomputed_contraction_summary.csv" if args.paper else "smoke_contraction_summary.csv"
    with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
