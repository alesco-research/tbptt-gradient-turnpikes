"""Saved-data contract checks for the inexpensive paper-figure renderer."""

import csv
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from gradient_turnpike.models import VanillaRNN


SCRIPT = Path(__file__).parents[1] / "experiments" / "make_paper_figures.py"
REPOSITORY_ROOT = SCRIPT.parent.parent
SPEC = spec_from_file_location("make_paper_figures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


CONFIG = {
    "fixed_parameter_gradient_analysis": {
        "past_contexts": [0, 5, 10, 20, 40], "future_contexts": [0, 5, 10, 20, 40],
        "interval_count": 20,
    },
    "reproducibility": {"seeds": [1, 2, 3, 4, 5]},
}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture_rows() -> list[dict]:
    aggregate = []
    contexts = CONFIG["fixed_parameter_gradient_analysis"]["past_contexts"]
    for future in contexts:
        for past in contexts:
            value = 100.0 * future + past + 1.0
            aggregate.append({"T_P": past, "T_F": future,
                              "median_absolute_error": value, "sample_count": 100})
    return aggregate


class HeatmapSavedDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.output_root = Path(self.directory.name) / "outputs"
        self.original_output_root = MODULE.OUTPUT_ROOT
        MODULE.OUTPUT_ROOT = self.output_root
        aggregate = fixture_rows()
        analysis = self.output_root / "analysis" / "gradient_error_heatmap"
        write_rows(analysis / "aggregated_gradient_errors.csv", aggregate)

    def tearDown(self):
        MODULE.OUTPUT_ROOT = self.original_output_root
        self.directory.cleanup()

    def test_matrix_is_reconstructed_from_complete_saved_data(self):
        matrix, past, future = MODULE.load_gradient_error_heatmap(CONFIG)
        self.assertEqual(past, [0, 5, 10, 20, 40])
        self.assertEqual(future, [0, 5, 10, 20, 40])
        self.assertEqual(matrix[0][0], 1.0)
        self.assertEqual(matrix[4][4], 4041.0)


class PointwiseAnalysisTests(unittest.TestCase):
    def test_reference_uses_complete_training_sequence(self):
        script = REPOSITORY_ROOT / "experiments" / "analyze_gradient_turnpike.py"
        spec = spec_from_file_location("analyze_gradient_turnpike", script)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(71)
        model = VanillaRNN(hidden_size=2).double()
        u = torch.linspace(-0.2, 0.3, 12, dtype=torch.float64).reshape(-1, 1)
        y = torch.zeros_like(u)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            checkpoint.touch()
            config = {
                "data": {"train_fraction": 0.8},
                "gradient_turnpike": {
                    "checkpoint": str(checkpoint), "update_start": 4,
                    "update_length": 3, "symmetric_contexts": [0, 2], "dtype": "float64",
                },
            }
            with (patch.object(module, "OUTPUT_ROOT", root),
                  patch.object(module, "load_config", return_value=config),
                  patch.object(module, "load_wh"),
                  patch.object(module, "load_model", return_value=(model, {"seed": 1})),
                  patch.object(module, "prepare_sequences", return_value=[(u, y)]),
                  patch.object(module, "runtime_metadata", return_value={}),
                  patch.object(module, "pointwise_gradient_contributions",
                               wraps=module.pointwise_gradient_contributions) as contributions,
                  patch("sys.argv", [str(script), "--device", "cpu"])):
                module.main()
            reference = contributions.call_args_list[0]
            self.assertIs(reference.args[1], u)
            self.assertIs(reference.args[2], y)
            self.assertEqual(reference.kwargs, {"a": 4, "b": 6})
            context_call = contributions.call_args_list[2]
            torch.testing.assert_close(context_call.args[1], u[2:9])
            self.assertEqual(context_call.kwargs, {"a": 2, "b": 4})
            metadata = json.loads((root / "analysis/gradient_turnpike/pointwise/config.json").read_text())
            self.assertEqual(metadata["full_context"], [0, 11])
            self.assertEqual((metadata["full_context_T_P"], metadata["full_context_T_F"]), (4, 5))


class VersionedAnalysisDataTests(unittest.TestCase):
    """Check the compact saved inputs used by the public figure renderer."""

    def test_pointwise_data_cover_each_symmetric_context(self):
        config = json.loads((REPOSITORY_ROOT / "configs" / "wh_paper.json").read_text())
        contexts = config["gradient_turnpike"]["symmetric_contexts"]
        path = (REPOSITORY_ROOT / "outputs" / "analysis" / "gradient_turnpike" / "pointwise"
                / "pointwise_gradient_errors.csv")
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertTrue(rows)
        metadata = json.loads((path.parent / "config.json").read_text())
        self.assertEqual(metadata["full_context"], [0, 79999])
        self.assertEqual(metadata["update_interval"], [40000, 40299])
        for context in contexts:
            values = [row for row in rows if int(row["T_P"]) == context and int(row["T_F"]) == context]
            self.assertEqual(len(values), config["gradient_turnpike"]["update_length"])
            self.assertEqual(sorted(int(row["relative_t"]) for row in values),
                             list(range(config["gradient_turnpike"]["update_length"])))

    def test_gradient_error_data_are_complete_and_provenanced(self):
        analysis = REPOSITORY_ROOT / "outputs" / "analysis" / "gradient_error_heatmap"
        with (analysis / "aggregated_gradient_errors.csv").open(newline="", encoding="utf-8") as file:
            aggregate = list(csv.DictReader(file))
        metadata = json.loads((analysis / "config.json").read_text())

        config = json.loads((REPOSITORY_ROOT / "configs" / "wh_paper.json").read_text())
        protocol = config["fixed_parameter_gradient_analysis"]
        expected_cells = {(past, future) for past in protocol["past_contexts"]
                          for future in protocol["future_contexts"]}
        self.assertEqual({(int(row["T_P"]), int(row["T_F"])) for row in aggregate}, expected_cells)
        self.assertEqual(len(aggregate), 25)
        self.assertTrue(all(int(row["sample_count"]) == 100 for row in aggregate))
        self.assertEqual(len(metadata["analysis_windows"]), protocol["interval_count"])
        self.assertEqual(len(metadata["checkpoint_identifiers"]), len(config["reproducibility"]["seeds"]))


if __name__ == "__main__":
    unittest.main()
