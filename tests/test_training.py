"""Offline checks of time indexing, full BPTT gradients, and optimization."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from gradient_turnpike.data import SignalData
from gradient_turnpike.models import VanillaRNN
from gradient_turnpike.outputs import read_resume_checkpoint
from gradient_turnpike.simulation import simulate
from gradient_turnpike.training import train_bptt


class RNNTests(unittest.TestCase):
    def test_bptt_resume_at_learning_rate_boundary_matches_uninterrupted(self):
        train = SignalData(self.u.float(), (self.u * .7).float(), 1.0)
        validation = SignalData(self.u.float(), (self.u * .7).float(), 1.0)
        schedule = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        lr = lambda update: .01 if update <= 4 else (.003 if update <= 8 else .001)
        initial = deepcopy(self.model)
        whole = deepcopy(initial)
        kwargs = dict(update_length=5, max_updates=12, validation_every_updates=1,
                      update_starts=schedule, learning_rate_schedule=lr, verbose=False)
        with TemporaryDirectory() as whole_directory, TemporaryDirectory() as split_directory:
            metadata = {"update_length": 5}
            whole_history = train_bptt(whole, train, validation, **kwargs,
                                       resume_output_directory=Path(whole_directory), resume_metadata=metadata,
                                       resume_every_updates=(12,))
            whole_payload = read_resume_checkpoint(Path(whole_directory) / "resume_update_00012.pt")
            staged = deepcopy(initial)
            split_history = train_bptt(staged, train, validation, **kwargs, stop_after=4,
                                       resume_output_directory=Path(split_directory), resume_metadata=metadata,
                                       resume_every_updates=(4,))
            checkpoint = Path(split_directory) / "resume_update_00004.pt"
            resumed = deepcopy(initial)
            resumed_history = train_bptt(resumed, train, validation, **kwargs,
                                         resume_checkpoint=str(checkpoint), resume_metadata=metadata,
                                         resume_output_directory=Path(split_directory),
                                         resume_every_updates=(12,))
            resumed_payload = read_resume_checkpoint(Path(split_directory) / "resume_update_00012.pt")
        for expected, actual in zip(whole.parameters(), resumed.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)
        torch.testing.assert_close(whole_payload["optimizer_state_dict"], resumed_payload["optimizer_state_dict"],
                                   rtol=0, atol=1e-14)
        torch.testing.assert_close(whole_payload["best_model_state_dict"], resumed_payload["best_model_state_dict"],
                                   rtol=0, atol=1e-14)
        self.assertEqual(whole_payload["best_validation_update"], resumed_payload["best_validation_update"])
        self.assertEqual(whole_payload["best_validation_loss"], resumed_payload["best_validation_loss"])
        self.assertEqual([row["update"] for row in resumed_history], list(range(1, 13)))
        self.assertEqual([{key: value for key, value in row.items()
                           if key not in {"wall_clock_time", "training_compute_seconds"}}
                          for row in whole_history],
                         [{key: value for key, value in row.items()
                           if key not in {"wall_clock_time", "training_compute_seconds"}}
                          for row in resumed_history])
        self.assertEqual(resumed_history[:4], split_history)
        self.assertGreater(resumed_history[-1]["training_compute_seconds"], split_history[-1]["training_compute_seconds"])
        self.assertGreater(resumed_history[-1]["wall_clock_time"], split_history[-1]["wall_clock_time"])
    def setUp(self):
        torch.manual_seed(7)
        self.model = VanillaRNN(hidden_size=2).double()
        self.u = torch.randn(8, 1, dtype=torch.float64) * 0.1

    def test_simulation_matches_manual_recurrence_and_continuation(self):
        model = self.model
        x0 = torch.tensor([0.2, -0.1], dtype=torch.float64)
        y, states = simulate(model, self.u, x0)
        x = x0
        expected_y, expected_states = [], [x0]
        for u in self.u:
            expected_y.append(model.readout(torch.cat((x, u))))
            x = torch.tanh(model.rnn.weight_ih_l0 @ u + model.rnn.bias_ih_l0
                           + model.rnn.weight_hh_l0 @ x + model.rnn.bias_hh_l0)
            expected_states.append(x)
        torch.testing.assert_close(y, torch.stack(expected_y))
        torch.testing.assert_close(states, torch.stack(expected_states))
        next_x, first_y = model(x0, self.u[0])
        torch.testing.assert_close(next_x, states[1])
        torch.testing.assert_close(first_y, y[0])
        first_y, first_states = simulate(model, self.u[:3], x0)
        last_y, _ = simulate(model, self.u[3:], first_states[-1])
        torch.testing.assert_close(torch.cat((first_y, last_y)), y)

    def test_input_and_initial_state_gradcheck(self):
        u = self.u.clone().requires_grad_()
        x0 = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda inputs, state: simulate(self.model, inputs, state), (u, x0)))

    def test_parameter_gradients_match_finite_differences(self):
        # Loss only at the last output: gradients must traverse earlier states.
        def loss():
            y, _ = simulate(self.model, self.u)
            return (y[-1] - 0.3).square().sum()

        loss().backward()
        epsilon = 1e-6
        for name, parameter in self.model.named_parameters():
            numerical = torch.zeros_like(parameter)
            with torch.no_grad():
                for i in range(parameter.numel()):
                    flat = parameter.view(-1)
                    original = flat[i].item()
                    flat[i] = original + epsilon
                    plus = loss().item()
                    flat[i] = original - epsilon
                    minus = loss().item()
                    flat[i] = original
                    numerical.view(-1)[i] = (plus - minus) / (2 * epsilon)
            with self.subTest(parameter=name):
                torch.testing.assert_close(parameter.grad, numerical, atol=1e-7, rtol=1e-5)

    def test_late_loss_reaches_first_input(self):
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.fill_(0.2)
        u = self.u.clone().requires_grad_()
        simulate(self.model, u)[0][-1].sum().backward()
        self.assertGreater(abs(u.grad[0].item()), 1e-8)

    def test_training_reduces_loss_and_restores_validation_minimum(self):
        # A deterministic, easy identification problem with no downloads.
        u = torch.linspace(-0.5, 0.5, 40).reshape(-1, 1)
        y = 0.5 * u + 0.1
        train = SignalData(u[:30], y[:30], 1.0)
        validation = SignalData(u[30:], y[30:], 1.0)
        history = train_bptt(self.model, train, validation, update_length=30, max_updates=50,
                             validation_every_updates=1, update_starts=[0] * 50,
                             learning_rate=0.02, verbose=False)
        self.assertLess(history[-1]["train_loss"], history[0]["train_loss"])
        self.assertFalse(self.model.training)
        with torch.no_grad():
            prediction, _ = simulate(self.model, u.double())
            mse = (prediction[30:] - y[30:].double()).square().mean().item()
        self.assertAlmostEqual(mse, min(row["val_loss"] for row in history), places=12)

    def test_epoch_losses_match_mse_and_validation_does_not_update(self):
        train = SignalData(self.u[:5], 0.5 * self.u[:5], 1.0)
        validation = SignalData(self.u[5:], 0.5 * self.u[5:], 1.0)
        reference = deepcopy(self.model)
        optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3)
        prediction, _ = simulate(reference, train.u)
        train_loss = (prediction - train.y).square().mean()
        train_loss.backward()
        optimizer.step()
        with torch.no_grad():
            _, states = simulate(reference, train.u)
            prediction, _ = simulate(reference, validation.u, states[-1])
            val_loss = (prediction - validation.y).square().mean().item()

        simulation_modes = []

        def record_simulation(model, *args):
            simulation_modes.append((model.training, torch.is_grad_enabled()))
            return simulate(model, *args)

        with patch("gradient_turnpike.training.simulate", side_effect=record_simulation):
            history = train_bptt(self.model, train, validation, update_length=5, max_updates=1,
                                 validation_every_updates=1, update_starts=[0], verbose=False)
        # The full-context gradient routine owns its simulator; validation still
        # uses exactly two no-grad calls through this module.
        self.assertEqual(simulation_modes, [(False, False), (False, False)])
        self.assertAlmostEqual(history[0]["train_loss"], train_loss.item(), places=12)
        self.assertAlmostEqual(history[0]["val_loss"], val_loss, places=12)
        for actual, expected in zip(self.model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)

    def test_epoch_callback_captures_final_weights_before_best_restoration(self):
        with torch.no_grad():
            initial_prediction, _ = simulate(self.model, self.u)
        train = SignalData(self.u[:5], initial_prediction[:5] + 1, 1.0)
        validation = SignalData(self.u[5:], initial_prediction[5:], 1.0)
        snapshots = []

        def capture_update(model, record):
            snapshots.append(deepcopy(model.state_dict()))

        history = train_bptt(self.model, train, validation, update_length=5, max_updates=5,
                             validation_every_updates=1, update_starts=[0] * 5,
                             verbose=False, on_update=capture_update)
        best_update = min(history, key=lambda row: row["val_loss"])["update"]
        self.assertEqual(len(snapshots), 5)
        self.assertLess(best_update, 5)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, snapshots[best_update - 1][name])
        self.assertTrue(any(not torch.equal(value, snapshots[-1][name])
                            for name, value in self.model.state_dict().items()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_split_calls_preserve_full_sequence_gradients(self):
        # Force small call boundaries so gradients across them remain measurable.
        # Compare float64 CUDA results to the unsplit CPU reference.
        reference = deepcopy(self.model)
        cpu_u = self.u.clone().requires_grad_()
        cpu_x0 = torch.tensor([0.2, -0.1], dtype=torch.float64, requires_grad=True)
        expected_y, expected_states = simulate(reference, cpu_u, cpu_x0)
        expected_y[-1].square().sum().backward()

        self.model.cuda()
        cuda_u = self.u.cuda().requires_grad_()
        cuda_x0 = cpu_x0.detach().cuda().requires_grad_()
        with patch("gradient_turnpike.simulation.CUDNN_MAX_SEQUENCE_LENGTH", 3):
            y, states = simulate(self.model, cuda_u, cuda_x0)
        y[-1].square().sum().backward()
        torch.testing.assert_close(y.cpu(), expected_y)
        torch.testing.assert_close(states.cpu(), expected_states)
        torch.testing.assert_close(cuda_u.grad.cpu(), cpu_u.grad)
        torch.testing.assert_close(cuda_x0.grad.cpu(), cpu_x0.grad)
        self.assertGreater(abs(cuda_u.grad[0].item()), 1e-10)
        for actual, expected in zip(self.model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad.cpu(), expected.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_sequence_exceeding_cudnn_limit(self):
        model = VanillaRNN(hidden_size=2).cuda()
        u = torch.zeros(65_536, 1, device="cuda")
        y, states = simulate(model, u)
        self.assertEqual(y.shape, (65_536, 1))
        self.assertEqual(states.shape, (65_537, 2))
        y.square().mean().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_forward_backward_and_training(self):
        cpu_y, _ = simulate(self.model, self.u)
        cpu_y.square().mean().backward()
        expected_gradients = [p.grad.clone() for p in self.model.parameters()]
        self.model.to("cuda")
        self.model.zero_grad(set_to_none=True)
        cuda_y, _ = simulate(self.model, self.u.cuda())
        cuda_y.square().mean().backward()
        torch.testing.assert_close(cuda_y.cpu(), cpu_y)
        for parameter, expected in zip(self.model.parameters(), expected_gradients):
            torch.testing.assert_close(parameter.grad.cpu(), expected)
        train = SignalData(self.u[:5].float(), self.u[:5].float(), 1.0)
        validation = SignalData(self.u[5:].float(), self.u[5:].float(), 1.0)
        history = train_bptt(self.model, train, validation, update_length=5, max_updates=1,
                             validation_every_updates=1, update_starts=[0], verbose=False)
        self.assertEqual(len(history), 1)
        self.assertEqual(next(self.model.parameters()).device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
