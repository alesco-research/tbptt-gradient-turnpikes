"""Output helpers work with any supplied run directory, without downloads."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch

from gradient_turnpike.outputs import plot_history, save_checkpoint, save_config, save_history


class OutputTests(unittest.TestCase):
    def test_history_retains_optimizer_progress_columns(self):
        record = dict(update=4, train_loss=0.2, val_loss=0.3, wall_clock_time=1.2,
                      processed_update_samples=200, processed_context_samples=360)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_history(output_dir, [record])
            with (output_dir / "training_history.csv").open(newline="") as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames[:3], ["update", "train_loss", "val_loss"])
                row = next(reader)
            self.assertEqual(int(row["update"]), 4)
            self.assertEqual(int(row["processed_update_samples"]), 200)

    def test_run_artifacts_and_checkpoint_roundtrip(self):
        records = [dict(update=1, train_loss=0.3, val_loss=0.4, wall_clock_time=0.1,
                       processed_update_samples=50, processed_context_samples=50),
                   dict(update=2, train_loss=0.2, val_loss=0.25, wall_clock_time=0.2,
                       processed_update_samples=100, processed_context_samples=100)]
        config = {"training_method": "example", "seed": 0}
        torch.manual_seed(config["seed"])
        model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "example" / "run_001"
            save_config(output_dir, config)
            save_history(output_dir, records)
            plot_history(output_dir, records)
            path = save_checkpoint(output_dir, "final_model.pt", model, config)
            with (output_dir / "training_history.csv").open(newline="") as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames[:3], ["update", "train_loss", "val_loss"])
                rows = list(reader)
            self.assertEqual([float(row["val_loss"]) for row in rows], [0.4, 0.25])
            self.assertEqual(json.loads((output_dir / "history.json").read_text()), records)
            self.assertEqual(json.loads((output_dir / "config.json").read_text()), config)
            self.assertEqual(json.loads((output_dir / "resolved_config.json").read_text()), config)
            self.assertTrue((output_dir / "validation_loss_vs_updates.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            checkpoint = torch.load(path, weights_only=True)
            self.assertEqual(checkpoint["metadata"], config)
            restored = torch.nn.Linear(1, 1)
            restored.load_state_dict(checkpoint["model_state_dict"])
            torch.testing.assert_close(restored(torch.ones(1, 1)), model(torch.ones(1, 1)))


if __name__ == "__main__":
    unittest.main()
