"""Offline contracts for the official Wiener--Hammerstein RMSE convention."""

import math
import unittest

import numpy as np
import torch
from nonlinear_benchmarks.error_metrics import RMSE
from nonlinear_benchmarks.utilities import Input_output_data

from gradient_turnpike.evaluation import official_wh_rmse


class OfficialWHRMSETests(unittest.TestCase):
    def test_training_style_full_trajectory_matches_manual_rmse(self):
        target = np.array([[1.0], [-2.0], [0.5], [3.0]], dtype=np.float32)
        prediction = torch.tensor([[0.5], [-1.0], [1.0], [2.0]], dtype=torch.float32)
        manual = math.sqrt(np.mean((prediction.numpy() - target) ** 2))
        self.assertAlmostEqual(official_wh_rmse(target, prediction), manual, places=7)
        self.assertAlmostEqual(float(np.asarray(RMSE(target, prediction.numpy())).squeeze()), manual, places=7)

    def test_test_style_uses_official_initialization_window(self):
        target = np.arange(55, dtype=np.float32).reshape(-1, 1)
        prediction = torch.from_numpy(target.copy())
        prediction[:50] += 100.0
        prediction[50:] += 2.0
        official_test = Input_output_data(
            u=np.zeros_like(target), y=target, state_initialization_window_length=50)
        manual = math.sqrt(np.mean((prediction.numpy()[50:] - target[50:]) ** 2))
        self.assertAlmostEqual(official_wh_rmse(official_test, prediction), manual, places=7)
        self.assertAlmostEqual(float(np.asarray(RMSE(official_test, prediction.numpy())).squeeze()), manual, places=7)


if __name__ == "__main__":
    unittest.main()
