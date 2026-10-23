"""Offline tests: synthetic sequences replace benchmark downloads."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gradient_turnpike.data import SignalData, load_wh, split_train_validation


class DataTests(unittest.TestCase):
    def setUp(self):
        self.u = torch.arange(11, dtype=torch.float32).reshape(-1, 1)
        self.data = SignalData(self.u, -self.u, 0.01)

    def test_contiguous_split(self):
        train, validation = split_train_validation(self.data, 0.7)
        self.assertEqual(len(train.u), 7)
        self.assertEqual(len(validation.u), 4)
        for field in ("u", "y"):
            torch.testing.assert_close(
                torch.cat((getattr(train, field), getattr(validation, field))),
                getattr(self.data, field),
            )
        self.assertEqual(train.sampling_time, 0.01)
        self.assertEqual(validation.sampling_time, 0.01)
        train.u[0] = 99
        self.assertEqual(self.data.u[0].item(), 0)

    def test_invalid_splits(self):
        for fraction in (0, 1, -0.1, 1.1, float("nan"), 0.01):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                split_train_validation(self.data, fraction)
        for data in (SignalData(self.u[:1], self.u[:1], 1),
                     SignalData(self.u[:0], self.u[:0], 1),
                     SignalData(self.u, self.u[:-1], 1),
                     SignalData(self.u[:, 0], self.u, 1)):
            with self.assertRaises(ValueError):
                split_train_validation(data)

    @patch("gradient_turnpike.data.nonlinear_benchmarks.WienerHammerBenchMark")
    def test_invalid_fraction_does_not_download(self, loader):
        with self.assertRaises(ValueError):
            load_wh(1.0)
        loader.assert_not_called()

    @patch("gradient_turnpike.data.nonlinear_benchmarks.WienerHammerBenchMark")
    def test_loading_preserves_test_and_converts_dtype(self, loader):
        development = SimpleNamespace(u=self.u.double().numpy(),
                                      y=(-self.u).double().numpy(), sampling_time=0.01)
        test_u = torch.arange(100, 106, dtype=torch.float64).reshape(-1, 1)
        official_test = SimpleNamespace(u=test_u.numpy(), y=(-test_u).numpy(),
                                        sampling_time=0.02, state_initialization_window_length=2)
        loader.return_value = development, official_test
        result = load_wh(0.7)
        loader.assert_called_once_with(atleast_2d=True)
        for sequence in (result.train, result.validation, result.test):
            for signal in (sequence.u, sequence.y):
                self.assertEqual(signal.dtype, torch.float32)
                self.assertEqual(signal.device.type, "cpu")
                self.assertEqual(signal.ndim, 2)
        torch.testing.assert_close(result.train.u, self.u[:7])
        torch.testing.assert_close(result.validation.y, -self.u[7:])
        torch.testing.assert_close(result.test.u, test_u.float())
        torch.testing.assert_close(result.test.y, -test_u.float())
        self.assertEqual(result.test.sampling_time, 0.02)
        self.assertEqual(result.test.state_initialization_window_length, 2)
        self.assertIs(result.official_test, official_test)
        result.test.u[0] = 0
        self.assertEqual(official_test.u[0, 0], 100)


if __name__ == "__main__":
    unittest.main()
