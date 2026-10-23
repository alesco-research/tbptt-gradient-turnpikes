"""Contracts for the compact public uniform-state contraction result."""

import csv
import json
import math
from pathlib import Path
import unittest

import torch

from gradient_turnpike.models import VanillaRNN
from gradient_turnpike.state_transitions import admissible_start_indices, analytic_state_jacobians, paper_search_settings, projected_spectral_ascent, transition_product_norms


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIRECTORY = ROOT / "outputs" / "analysis" / "uniform_state_contraction"


class StateTransitionJacobianTests(unittest.TestCase):
    def test_admissible_starts_include_exactly_every_length_t_slice(self):
        starts = admissible_start_indices(7, 3)
        self.assertEqual(starts.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(int(starts[0]), 0)
        self.assertEqual(int(starts[-1]), 4)
        self.assertEqual(len(starts), 5)
        self.assertNotIn(5, starts.tolist())
    def test_analytic_tanh_state_jacobian_matches_autograd(self):
        torch.manual_seed(701)
        model = VanillaRNN().to(dtype=torch.float64)
        x = torch.randn(16, dtype=torch.float64)
        u = torch.randn(1, dtype=torch.float64)
        with torch.no_grad():
            x_next = model.f(x, u)
        analytic = analytic_state_jacobians(torch.stack((x, x_next)), model.rnn.weight_hh_l0)[0]
        autograd = torch.autograd.functional.jacobian(lambda state: model.f(state, u), x)
        torch.testing.assert_close(analytic, autograd, rtol=1e-12, atol=1e-12)

    def test_multistep_product_matches_autograd_and_projection(self):
        torch.manual_seed(19)
        model = VanillaRNN().to(dtype=torch.float64)
        a, b = model.rnn.weight_hh_l0, model.rnn.weight_ih_l0
        bias = model.rnn.bias_ih_l0 + model.rnn.bias_hh_l0
        inputs = torch.randn(5, 1, dtype=torch.float64)
        x = torch.full((1, 16), 2.0, dtype=torch.float64)
        starts = torch.tensor([1])
        values, states = projected_spectral_ascent(a, b, bias, inputs, starts, x, 3, iterations=3, learning_rate=.1)
        self.assertEqual(values.dtype, torch.float64)
        self.assertTrue(bool(((states >= -1) & (states <= 1)).all()))
        initial = states[0]
        def flow(state):
            for u in inputs[1:4]:
                state = model.f(state, u)
            return state
        exact = torch.linalg.svdvals(torch.autograd.functional.jacobian(flow, initial))[0]
        measured = transition_product_norms(a, b, bias, inputs, starts, states, 3)[0]
        torch.testing.assert_close(measured, exact, rtol=1e-11, atol=1e-11)


class StateTransitionContractionDataTests(unittest.TestCase):
    def test_saved_contraction_summary_contract(self):
        with (ANALYSIS_DIRECTORY / "contraction_summary.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 5)
        self.assertEqual([int(row["seed"]) for row in rows], [1, 2, 3, 4, 5])
        self.assertEqual(len({row["seed"] for row in rows}), 5)
        for row in rows:
            self.assertEqual(int(row["checkpoint_update"]), 5000)
            self.assertEqual(int(row["state_dim"]), 16)
            self.assertEqual(int(row["trajectory_length"]), 100000)
            self.assertEqual(int(row["train_length"]), 80000)
            self.assertEqual(int(row["validation_length"]), 20000)
            self.assertEqual(int(row["horizon"]), 20)
            self.assertLess(float(row["worst_value_found"]), 1.0)
            self.assertEqual(row["numerical_counterexample_found"], "False")

    def test_saved_metadata_contract(self):
        metadata = json.loads((ANALYSIS_DIRECTORY / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["horizon"], 20)
        self.assertEqual(metadata["dtype"], "float64")
        self.assertEqual(metadata["checkpoint_update"], 5000)
        self.assertEqual(metadata["state_dimension"], 16)
        self.assertEqual(metadata["trajectory"]["trajectory_length"], 100000)
        self.assertEqual(metadata["trajectory"]["train_length"], 80000)
        self.assertEqual(metadata["trajectory"]["validation_length"], 20000)
        self.assertFalse(metadata["trajectory"]["official_test_used"])

    def test_summary_maximum_and_seed_specific_paper_settings(self):
        with (ANALYSIS_DIRECTORY / "contraction_summary.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        expected = max(rows, key=lambda row: float(row["worst_value_found"]))
        summary = json.loads((ANALYSIS_DIRECTORY / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(float(expected["worst_value_found"]), summary["overall_maximum"])
        self.assertEqual(int(expected["seed"]), summary["overall_maximum_seed"])
        seed_three = paper_search_settings(3)
        self.assertEqual((seed_three["candidate_windows"], seed_three["final_windows"]), (1000, 150))
        self.assertEqual((seed_three["first_iterations"], seed_three["final_iterations"]), (75, 250))
        self.assertEqual(seed_three["learning_rates"], (0.01, 0.03, 0.06))
        for seed in (1, 2, 4, 5):
            self.assertEqual(paper_search_settings(seed)["candidate_windows"], 300)


if __name__ == "__main__":
    unittest.main()
