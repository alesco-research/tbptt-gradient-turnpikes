"""Lightweight contracts for the compact final training-comparison output."""

import csv
import importlib.util
import json
import math
from pathlib import Path
from statistics import median
import unittest

from gradient_turnpike.metrics import augment_paired_bptt_metrics, paired_ratio_median, summarize_training_comparison


ROOT = Path("outputs/analysis/training_comparison")


def comparison_module():
    spec = importlib.util.spec_from_file_location(
        "training_comparison", "experiments/run_training_comparison.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TrainingComparisonTests(unittest.TestCase):
    def test_paired_gap_and_runtime_are_seed_matched(self):
        module = comparison_module()
        rows = [
            {"seed": 1, "method": "BPTT", "full_training_mse": .2,
             "training_time_seconds": 10.},
            {"seed": 1, "method": "TBPTT", "full_training_mse": .3,
             "training_time_seconds": 2.},
            {"seed": 2, "method": "BPTT", "full_training_mse": .4,
             "training_time_seconds": 20.},
            {"seed": 2, "method": "TBPTT", "full_training_mse": .5,
             "training_time_seconds": 5.},
        ]
        augment_paired_bptt_metrics(rows, expected_seeds=(1, 2))
        self.assertEqual(rows[0]["bptt_reference_loss_gap"], 0.)
        self.assertAlmostEqual(rows[1]["bptt_reference_loss_gap"], .1)
        self.assertAlmostEqual(rows[1]["relative_bptt_reference_loss_gap"], .5)
        self.assertAlmostEqual(rows[1]["speedup_vs_bptt"], 5.)
        self.assertAlmostEqual(rows[3]["bptt_reference_loss_gap"], .1)
        self.assertAlmostEqual(rows[3]["relative_bptt_reference_loss_gap"], .25)
        self.assertAlmostEqual(rows[3]["speedup_vs_bptt"], 4.)

    def test_saved_final_summary_reconstructs_from_per_run_rows(self):
        module = comparison_module()
        with (ROOT / "per_run_metrics.csv").open() as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 25)
        self.assertEqual(sum(row["method"] == "BPTT" for row in rows), 5)
        for seed in map(str, range(1, 6)):
            contexts = sorted(int(row["T_P"]) for row in rows
                              if row["seed"] == seed and row["method"] == "TBPTT")
            self.assertEqual(contexts, [0, 10, 20, 40])
        for row in rows:
            row["seed"] = int(row["seed"])
            for key in ("best_validation_mse", "full_training_mse", "full_training_rmse", "test_mse", "test_rmse",
                        "training_time_seconds", "bptt_reference_loss_gap",
                        "relative_bptt_reference_loss_gap", "runtime_relative_to_bptt", "speedup_vs_bptt"):
                row[key] = float(row[key])
        augment_paired_bptt_metrics(rows, expected_seeds=(1, 2, 3, 4, 5))
        summary, _ = summarize_training_comparison(rows)
        with (ROOT / "summary.csv").open(newline="") as file:
            saved = list(csv.DictReader(file))
        self.assertEqual(len(summary), len(saved))
        for actual, expected in zip(summary, saved):
            self.assertEqual(actual["method"], expected["method"])
            self.assertEqual(actual["T_P=T_F"], expected["T_P=T_F"])
            for key, value in actual.items():
                if key not in {"method", "T_P=T_F"}:
                    self.assertAlmostEqual(float(value), float(expected[key]))

    def test_paper_table_statistics_reconstruct_from_compact_per_run_data(self):
        """Train/test relative RMSE and speedup are medians of seed-paired ratios."""
        with (ROOT / "per_run_metrics.csv").open() as file:
            rows = list(csv.DictReader(file))
        bptt = {int(row["seed"]): row for row in rows if row["method"] == "BPTT"}
        self.assertEqual(set(bptt), {1, 2, 3, 4, 5})
        expected = {
            0: ("0.03950", "0.03850", "1.55", "1.58"),
            10: ("0.03307", "0.03248", "1.50", "1.51"),
            20: ("0.04591", "0.04527", "2.25", "2.25"),
            40: ("0.02691", "0.02622", "1.11", "1.12"),
        }
        self.assertEqual(
            (f"{median(float(row['full_training_rmse']) for row in bptt.values()):.5f}",
             f"{median(float(row['test_rmse']) for row in bptt.values()):.5f}"),
            ("0.02364", "0.02284"),
        )
        for context in (0, 10, 20, 40):
            group = {int(row["seed"]): row for row in rows
                     if row["method"] == "TBPTT" and int(row["T_P"]) == context}
            self.assertEqual(set(group), set(bptt))
            train_ratio = paired_ratio_median(
                {seed: float(row["full_training_rmse"]) for seed, row in group.items()},
                {seed: float(row["full_training_rmse"]) for seed, row in bptt.items()},
            )
            test_ratio = paired_ratio_median(
                {seed: float(row["test_rmse"]) for seed, row in group.items()},
                {seed: float(row["test_rmse"]) for seed, row in bptt.items()},
            )
            speedup = median(float(row["speedup_vs_bptt"]) for row in group.values())
            self.assertTrue(all(math.isfinite(value) and value > 0 for value in (train_ratio, test_ratio, speedup)))
            self.assertEqual(
                (f"{median(float(row['full_training_rmse']) for row in group.values()):.5f}",
                 f"{median(float(row['test_rmse']) for row in group.values()):.5f}",
                 f"{train_ratio:.2f}", f"{test_ratio:.2f}"),
                expected[context],
            )

    def test_paper_configuration_freezes_the_declared_protocol(self):
        module = comparison_module()
        config = json.loads((ROOT / "config.json").read_text())
        self.assertEqual(config["bptt"]["update_length"], 1000)
        self.assertEqual(config["bptt"]["mini_batch_size"], 1)
        self.assertEqual(config["tbptt"]["contexts"], [0, 10, 20, 40])
        self.assertEqual(config["tbptt"]["update_length"], 50)
        self.assertEqual(config["tbptt"]["mini_batch_size"], 4)
        self.assertEqual(config["tbptt"]["start_domain"], [40, 79910])
        self.assertEqual(config["validation_selection"]["cadence_updates"], 500)
        self.assertTrue(config["validation_selection"]["strict_improvement"])
        self.assertEqual(config["runtime_measurement"]["cuda_synchronization"],
                         "immediately before and after every optimizer update")

    def test_saved_runtime_provenance_is_semantic_and_protocol_matched(self):
        with (ROOT / "per_run_metrics.csv").open() as file:
            rows = list(csv.DictReader(file))
        self.assertNotIn("checkpoint_identifier", rows[0])
        for row in rows:
            self.assertIn(row["model_metrics_source"], {"frozen_production_run", "production_training_run"})
            self.assertIn(row["timing_source"], {
                "deterministic_protocol_timing_repeat", "synchronized_production_training_compute"})
            self.assertNotIn("outputs/", row["model_metrics_source"])
            self.assertNotIn("outputs/", row["timing_source"])
        bptt = {int(row["seed"]): row for row in rows if row["method"] == "BPTT"}
        self.assertEqual({seed for seed, row in bptt.items()
                          if row["model_metrics_source"] == "frozen_production_run"}, {1, 2})
        self.assertEqual({seed for seed, row in bptt.items()
                          if row["timing_source"] == "deterministic_protocol_timing_repeat"}, {1, 2})


if __name__ == "__main__":
    unittest.main()
