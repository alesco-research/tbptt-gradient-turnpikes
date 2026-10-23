"""Float64 checks of occurrence-restricted gradients, using offline toy data."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from torch.func import functional_call

from gradient_turnpike.data import SignalData
from gradient_turnpike.gradients import (
    ContextWindow, admissible_update_start_bounds, make_context_window, make_disjoint_context_windows,
    flatten_parameter_gradient, minibatch_tbptt_loss, pointwise_gradient_contributions,
    sample_update_minibatches, sample_update_starts, tbptt_loss, validate_context_window,
)
from gradient_turnpike.models import VanillaRNN
from gradient_turnpike.outputs import read_resume_checkpoint
from gradient_turnpike.simulation import simulate
from gradient_turnpike.tbptt import train_tbptt
from gradient_turnpike.training import evaluate_validation, train_bptt


def parameter_gradients(loss, model):
    parameters = tuple(model.parameters())
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, grads)]


class TBPTTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = VanillaRNN(hidden_size=2, output_size=2).double()
        self.u = torch.linspace(-0.3, 0.4, 7, dtype=torch.float64).reshape(-1, 1)
        self.y = torch.cat((0.2 + self.u, -0.1 - self.u), dim=1)

    def assert_gradients_close(self, actual, expected):
        for p, a, e in zip(self.model.parameters(), actual, expected):
            self.assertEqual(a.shape, p.shape)
            self.assertTrue(torch.isfinite(a).all())
            torch.testing.assert_close(a, e, atol=1e-10, rtol=1e-8)

    def objective(self, model=None, y=None, T_P=2, T_F=2, update_length=3):
        n = T_P + update_length + T_F
        return tbptt_loss(self.model if model is None else model, self.u[:n],
                          self.y[:n] if y is None else y, T_P=T_P, T_F=T_F,
                          update_length=update_length)[0]

    def test_windows_have_exact_context_and_omit_partial_updates(self):
        self.assertEqual(make_disjoint_context_windows(18, 2, 3, 4),
                         [ContextWindow(0, 2, 5, 8), ContextWindow(4, 6, 9, 12),
                          ContextWindow(8, 10, 13, 16)])
        self.assertEqual(make_disjoint_context_windows(7, 0, 0, 7), [ContextWindow(0, 0, 6, 6)])
        for args in ((7, -1, 0, 2), (7, 0, -1, 2), (7, 0, 0, 0), (7, 3, 3, 2)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                make_disjoint_context_windows(*args)
        with self.assertRaises(ValueError):
            tbptt_loss(self.model, self.u[:-1], self.y[:-1], T_P=2, T_F=2, update_length=3)

    def test_context_window_maps_inclusive_math_to_python_slice(self):
        window = make_context_window(update_start=40, update_length=50, past_context=10, future_context=20)
        self.assertEqual(window, ContextWindow(30, 40, 89, 109))
        self.assertEqual(window.context_length, 80)
        self.assertEqual(window.update_length, 50)
        self.assertEqual(window.context_slice, slice(30, 110))
        self.assertEqual(admissible_update_start_bounds(100, update_length=50, past_context=10,
                                                        future_context=20), (10, 30))
        validate_context_window(window, 110)
        with self.assertRaises(ValueError):
            validate_context_window(window, 109)

    def test_shared_update_schedule_is_deterministic_and_respects_grid_margin(self):
        first = sample_update_starts(100, update_length=8, max_updates=6, seed=4,
                                     past_margin=20, future_margin=20)
        self.assertEqual(first, sample_update_starts(100, update_length=8, max_updates=6, seed=4,
                                                      past_margin=20, future_margin=20))
        self.assertTrue(all(20 <= a and a + 8 + 20 <= 100 for a in first))
        self.assertNotEqual(first, sample_update_starts(100, update_length=8, max_updates=6, seed=5,
                                                         past_margin=20, future_margin=20))

    def test_tbptt_minibatch_schedule_and_objective_are_exact_means(self):
        first = sample_update_minibatches(55, update_length=5, max_updates=4, mini_batch_size=3, seed=8,
                                          past_margin=4, future_margin=4)
        self.assertEqual(first, sample_update_minibatches(55, update_length=5, max_updates=4, mini_batch_size=3,
                                                           seed=8, past_margin=4, future_margin=4))
        self.assertTrue(all(len(batch) == 3 for batch in first))
        self.assertTrue(all(4 <= a <= 46 for batch in first for a in batch))
        self.assertTrue(any(len(set(batch)) < len(batch) for batch in
                            sample_update_minibatches(10, update_length=5, max_updates=2, mini_batch_size=4, seed=1)))
        torch.manual_seed(43)
        model = VanillaRNN(hidden_size=3).to(dtype=torch.float64)
        u = torch.randn(12, 1, dtype=torch.float64)
        y = torch.randn(12, 1, dtype=torch.float64)
        starts = [2, 2, 5]
        objective, _ = minibatch_tbptt_loss(model, u, y, starts=starts, T_P=1, T_F=1, update_length=2)
        combined = flatten_parameter_gradient(model, objective)
        explicit = torch.stack([
            flatten_parameter_gradient(model, tbptt_loss(
                model, u[a - 1:a + 3], y[a - 1:a + 3], T_P=1, T_F=1, update_length=2)[0])
            for a in starts
        ]).mean(0)
        torch.testing.assert_close(combined, explicit, rtol=1e-10, atol=1e-11)

    def test_production_gradient_matches_independent_reference_TP3_TF2_U4(self):
        """Compare the production surrogate with independent per-time copies."""
        torch.manual_seed(23)
        model = VanillaRNN(hidden_size=3).double()
        u = torch.randn(9, 1, dtype=torch.float64)
        y = torch.randn(9, 1, dtype=torch.float64)
        T_P, T_F, update_length = 3, 2, 4
        objective, _ = tbptt_loss(model, u, y, T_P=T_P, T_F=T_F, update_length=update_length)
        actual = parameter_gradients(objective, model)

        # A separate parameter copy at every time index is a direct occurrence
        # reference, independent of tbptt_loss and its functional-call segments.
        copies = [{name: value.detach().clone().requires_grad_()
                   for name, value in model.named_parameters()} for _ in range(len(u))]
        state = torch.zeros(3, dtype=torch.float64)
        reference_loss = torch.zeros((), dtype=torch.float64)
        for time_index in range(len(u)):
            state, prediction = functional_call(model, copies[time_index],
                                                (state, u[time_index]), strict=True)
            if time_index >= T_P:
                reference_loss = reference_loss + (prediction - y[time_index]).square().mean() / update_length
        all_copies = [value for at_time in copies for value in at_time.values()]
        gradients = torch.autograd.grad(reference_loss, all_copies, allow_unused=True)
        parameter_count = len(copies[0])
        expected = [sum((gradients[t * parameter_count + j] for t in range(T_P, T_P + update_length)),
                        torch.zeros_like(parameter))
                    for j, parameter in enumerate(model.parameters())]
        for gradient, parameter in zip(actual, model.parameters()):
            self.assertEqual(gradient.shape, parameter.shape)
            self.assertTrue(torch.isfinite(gradient).all())
        for actual_gradient, expected_gradient in zip(actual, expected):
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-11, rtol=1e-9)

    def test_pointwise_adjoint_contributions_sum_to_production_gradient(self):
        """The adjoint-form g_t values use the trainer's exact surrogate."""
        T_P, T_F, update_length = 2, 2, 3
        objective, _ = tbptt_loss(self.model, self.u, self.y, T_P=T_P, T_F=T_F,
                                  update_length=update_length)
        expected = torch.cat([value.reshape(-1) for value in parameter_gradients(objective, self.model)])
        contributions = pointwise_gradient_contributions(
            self.model, self.u, self.y, a=T_P, b=T_P + update_length - 1)
        self.assertEqual(len(contributions), update_length)
        for contribution in contributions:
            self.assertEqual(contribution.shape, expected.shape)
            self.assertTrue(torch.isfinite(contribution).all())
        actual = torch.stack(contributions).sum(dim=0) / update_length
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-9)

    def test_pointwise_full_context_matches_time_indexed_parameters(self):
        copies = [{name: p.detach().clone().requires_grad_()
                   for name, p in self.model.named_parameters()} for _ in self.u]
        state = self.u.new_zeros(2)
        loss = 0
        for t in range(len(self.u)):
            state, prediction = functional_call(self.model, copies[t], (state, self.u[t]), strict=True)
            loss = loss + (prediction - self.y[t]).square().mean()
        parameters = [p for at_time in copies for p in at_time.values()]
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        count = len(copies[0])
        for a, b in ((0, 0), (2, 4), (6, 6), (0, 6)):
            with self.subTest(a=a, b=b):
                actual = pointwise_gradient_contributions(self.model, self.u, self.y, a=a, b=b)
                for t, contribution in zip(range(a, b + 1), actual):
                    values = gradients[t * count:(t + 1) * count]
                    expected = torch.cat([
                        (torch.zeros_like(p) if g is None else g).reshape(-1)
                        for p, g in zip(self.model.parameters(), values)
                    ])
                    torch.testing.assert_close(contribution, expected, atol=1e-12, rtol=1e-10)

    def test_one_sided_equals_fixed_burn_in_then_update_bptt(self):
        for T_P in (0, 2):
            with self.subTest(T_P=T_P):
                actual = parameter_gradients(self.objective(T_P=T_P, T_F=0), self.model)
                x = self.u.new_zeros(2)
                if T_P:
                    with torch.no_grad():
                        _, states = simulate(self.model, self.u[:T_P])
                        x = states[-1]
                y, _ = simulate(self.model, self.u[T_P:T_P + 3], x)
                expected = parameter_gradients((y - self.y[T_P:T_P + 3]).square().mean(), self.model)
                self.assert_gradients_close(actual, expected)

    def test_full_interval_matches_full_bptt(self):
        actual = parameter_gradients(self.objective(T_P=0, T_F=0, update_length=7), self.model)
        prediction, _ = simulate(self.model, self.u)
        expected = parameter_gradients((prediction - self.y).square().mean(), self.model)
        self.assert_gradients_close(actual, expected)

    def test_only_update_time_parameter_occurrences_contribute(self):
        actual = parameter_gradients(self.objective(), self.model)
        # Independent reference: untie parameters across all seven time steps.
        copies = [{name: p.detach().clone().requires_grad_()
                   for name, p in self.model.named_parameters()} for _ in self.u]
        x = self.u.new_zeros(2)
        loss = 0
        for t, (u, target) in enumerate(zip(self.u, self.y)):
            x, prediction = functional_call(self.model, copies[t], (x, u), strict=True)
            loss = loss + (prediction - target).square().mean() / 3
        all_parameters = [p for at_time in copies for p in at_time.values()]
        gradients = torch.autograd.grad(loss, all_parameters, allow_unused=True)
        count = len(copies[0])
        expected = [sum((gradients[t * count + j] for t in range(2, 5)),
                        torch.zeros_like(p)) for j, p in enumerate(self.model.parameters())]
        self.assert_gradients_close(actual, expected)
        # Ordinary context gradients must differ from this restricted sum.
        prediction, _ = simulate(self.model, self.u)
        ordinary = parameter_gradients((prediction - self.y).square().sum() / 6, self.model)
        self.assertTrue(any(not torch.allclose(a, b) for a, b in zip(actual, ordinary)))

    def test_future_targets_change_dynamics_gradient_but_not_direct_readout_gradient(self):
        actual = parameter_gradients(self.objective(), self.model)
        changed = self.y.clone()
        changed[5:] += 2
        future_changed = parameter_gradients(self.objective(y=changed), self.model)
        self.assertTrue(any(not torch.allclose(a, b) for a, b in zip(actual[:4], future_changed[:4])))
        for a, b in zip(actual[4:], future_changed[4:]):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=0)
        one_sided = parameter_gradients(self.objective(T_F=0), self.model)
        self.assertTrue(any(not torch.allclose(a, b) for a, b in zip(actual, one_sided)))
        changed = self.y.clone()
        changed[:2] += 10
        self.assert_gradients_close(actual, parameter_gradients(self.objective(y=changed), self.model))

    def test_restricted_gradient_matches_finite_differences(self):
        actual = parameter_gradients(self.objective(), self.model)
        fixed = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        variable = {name: p.clone() for name, p in fixed.items()}

        def objective_with_frozen_context():
            x = self.u.new_zeros(2)
            result = 0
            for t in range(7):
                parameters = variable if 2 <= t <= 4 else fixed
                x, prediction = functional_call(self.model, parameters, (x, self.u[t]), strict=True)
                if t >= 2:
                    result = result + (prediction - self.y[t]).square().mean() / 3
            return result.item()

        numerical = []
        epsilon = 1e-6
        for value in variable.values():
            grad = torch.zeros_like(value)
            for i in range(value.numel()):
                flat = value.view(-1)
                original = flat[i].item()
                flat[i] = original + epsilon
                plus = objective_with_frozen_context()
                flat[i] = original - epsilon
                minus = objective_with_frozen_context()
                flat[i] = original
                grad.view(-1)[i] = (plus - minus) / (2 * epsilon)
            numerical.append(grad)
        self.assert_gradients_close(actual, numerical)

    def test_future_losses_reach_transition_at_last_update_index(self):
        # For |U|=1, y_a precedes f_a: only future losses can differentiate f_a.
        no_future = parameter_gradients(self.objective(T_F=0, update_length=1), self.model)
        with_future = parameter_gradients(self.objective(T_F=2, update_length=1), self.model)
        for grad in no_future[:4]:
            self.assertEqual(torch.count_nonzero(grad).item(), 0)
        self.assertTrue(any(torch.count_nonzero(grad).item() > 0 for grad in with_future[:4]))

    def test_full_window_training_matches_baseline_and_validation(self):
        baseline = deepcopy(self.model)
        train = SignalData(self.u[:5], self.y[:5], 1)
        validation = SignalData(self.u[5:], self.y[5:], 1)
        expected = train_bptt(baseline, train, validation, update_length=5, max_updates=3,
                              validation_every_updates=1, update_starts=[0, 0, 0], verbose=False)
        history = train_tbptt(self.model, train, validation, T_P=0, T_F=0,
                              update_length=5, max_updates=3, validation_every_updates=1,
                              update_starts=[0, 0, 0], verbose=False)
        for row, other in zip(history, expected):
            self.assertAlmostEqual(row['train_loss'], other['train_loss'], places=12)
            self.assertAlmostEqual(row['val_loss'], other['val_loss'], places=12)
        for a, b in zip(self.model.parameters(), baseline.parameters()):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=0)
        value = evaluate_validation(self.model, train.u, validation.u, validation.y)
        self.assertAlmostEqual(value, min(row['val_loss'] for row in history), places=12)

    def test_two_sided_training_counters_and_reproducible_shuffle(self):
        other = deepcopy(self.model)
        train = SignalData(self.u, self.y, 1)
        validation = SignalData(self.u[:2], self.y[:2], 1)
        kwargs = dict(T_P=1, T_F=1, update_length=2, max_updates=2,
                      validation_every_updates=1, update_starts=[1, 3], seed=10, verbose=False)
        history = train_tbptt(self.model, train, validation, **kwargs)
        repeated = train_tbptt(other, train, validation, **kwargs)
        for first, second in zip(history, repeated):
            self.assertEqual({key: value for key, value in first.items()
                              if key not in {"wall_clock_time", "training_compute_seconds"}},
                             {key: value for key, value in second.items()
                              if key not in {"wall_clock_time", "training_compute_seconds"}})
            self.assertGreaterEqual(first["wall_clock_time"], 0)
            self.assertGreaterEqual(second["wall_clock_time"], 0)
        self.assertEqual(history[-1]['update'], 2)
        self.assertEqual(history[-1]['processed_update_samples'], 4)
        self.assertEqual(history[-1]['processed_context_samples'], 8)

    def test_tbptt_resume_at_learning_rate_boundary_matches_uninterrupted(self):
        train = SignalData(self.u.float(), self.y.float(), 1.0)
        validation = SignalData(self.u.float(), self.y.float(), 1.0)
        schedule = [[1, 1], [2, 3], [1, 3], [2, 2], [1, 1], [2, 3],
                    [1, 3], [2, 2], [1, 1], [2, 3], [1, 3], [2, 2]]
        lr = lambda update: .01 if update <= 4 else (.003 if update <= 8 else .001)
        initial = deepcopy(self.model)
        whole = deepcopy(initial)
        kwargs = dict(T_P=1, T_F=1, update_length=3, max_updates=12,
                      validation_every_updates=1, update_starts=schedule,
                      learning_rate_schedule=lr, verbose=False)
        metadata = {"T_P": 1, "T_F": 1, "update_length": 3, "max_updates": 12}
        with TemporaryDirectory() as whole_directory, TemporaryDirectory() as split_directory:
            whole_history = train_tbptt(whole, train, validation, **kwargs,
                                        resume_output_directory=Path(whole_directory), resume_metadata=metadata,
                                        resume_every_updates=(12,))
            whole_payload = read_resume_checkpoint(Path(whole_directory) / "resume_update_00012.pt")
            staged = deepcopy(initial)
            train_tbptt(staged, train, validation, **kwargs, stop_after=4,
                        resume_output_directory=Path(split_directory), resume_metadata=metadata,
                        resume_every_updates=(4,))
            checkpoint = Path(split_directory) / "resume_update_00004.pt"
            resumed = deepcopy(initial)
            split_history = read_resume_checkpoint(checkpoint)["history"]
            resumed_history = train_tbptt(resumed, train, validation, **kwargs,
                                         resume_checkpoint=str(checkpoint), resume_metadata=metadata,
                                         resume_output_directory=Path(split_directory), resume_every_updates=(12,))
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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_two_sided_gradient_and_training(self):
        expected = parameter_gradients(self.objective(), self.model)
        self.model.cuda()
        objective, _ = tbptt_loss(self.model, self.u.cuda(), self.y.cuda(),
                                  T_P=2, T_F=2, update_length=3)
        actual = [g.cpu() for g in parameter_gradients(objective, self.model)]
        self.assert_gradients_close(actual, expected)
        contributions = pointwise_gradient_contributions(
            self.model, self.u.cuda(), self.y.cuda(), a=2, b=4)
        pointwise_mean = torch.stack(contributions).mean(0).cpu()
        torch.testing.assert_close(pointwise_mean, torch.cat([g.reshape(-1) for g in expected]),
                                   atol=1e-10, rtol=1e-8)
        train = SignalData(self.u, self.y, 1)
        validation = SignalData(self.u[:2], self.y[:2], 1)
        history = train_tbptt(self.model, train, validation, T_P=2, T_F=2,
                              update_length=3, max_updates=2, validation_every_updates=1,
                              update_starts=[2, 2], verbose=False)
        self.assertEqual(history[-1]['update'], 2)
        self.assertEqual(next(self.model.parameters()).device.type, 'cuda')


if __name__ == '__main__':
    unittest.main()
