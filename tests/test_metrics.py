"""Tests for seed-paired reporting semantics."""

import unittest

from gradient_turnpike.metrics import augment_paired_bptt_metrics, paired_ratio_median


class PairedMetricTests(unittest.TestCase):
    def test_seed_pairing_precedes_aggregation(self):
        rows = [
            {"seed": 1, "method": "BPTT", "full_training_mse": .25, "training_time_seconds": 10.},
            {"seed": 2, "method": "BPTT", "full_training_mse": 1., "training_time_seconds": 20.},
            {"seed": 1, "method": "TBPTT", "full_training_mse": 1., "training_time_seconds": 2.},
            {"seed": 2, "method": "TBPTT", "full_training_mse": 1., "training_time_seconds": 5.},
        ]
        augment_paired_bptt_metrics(rows, expected_seeds=(1, 2))
        self.assertEqual(rows[2]["speedup_vs_bptt"], 5.)
        self.assertEqual(rows[3]["speedup_vs_bptt"], 4.)
        self.assertEqual(paired_ratio_median({1: 2., 2: 1.}, {1: 1., 2: 2.}), 1.25)

    def test_missing_or_duplicate_bptt_references_fail_loudly(self):
        duplicate = [
            {"seed": 1, "method": "BPTT", "full_training_mse": 1., "training_time_seconds": 1.},
            {"seed": 1, "method": "BPTT", "full_training_mse": 1., "training_time_seconds": 1.},
        ]
        with self.assertRaises(ValueError):
            augment_paired_bptt_metrics(duplicate, expected_seeds=(1,))


if __name__ == "__main__":
    unittest.main()
