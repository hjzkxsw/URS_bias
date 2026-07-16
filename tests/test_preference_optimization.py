import math
import unittest

import torch

from Trainer import preference_optimization_loss


class PreferenceOptimizationLossTests(unittest.TestCase):
    def test_matches_bradley_terry_objective(self):
        rewards = torch.tensor([[2.0, 1.0]])
        log_probs = torch.tensor([[math.log(0.8), math.log(0.2)]])

        loss = preference_optimization_loss(rewards, log_probs, alpha=1.0)

        expected = -math.log(0.8) / 4
        self.assertAlmostEqual(loss.item(), expected, places=6)

    def test_gradient_favors_the_preferred_route(self):
        rewards = torch.tensor([[2.0, 1.0]])
        log_probs = torch.zeros((1, 2), requires_grad=True)

        loss = preference_optimization_loss(rewards, log_probs, alpha=1.0)
        loss.backward()

        self.assertLess(log_probs.grad[0, 0].item(), 0)
        self.assertGreater(log_probs.grad[0, 1].item(), 0)

    def test_is_invariant_to_positive_reward_scaling(self):
        log_probs = torch.tensor([[-1.0, -2.0, -3.0]])
        rewards = torch.tensor([[3.0, 2.0, 1.0]])

        original = preference_optimization_loss(rewards, log_probs, alpha=0.03)
        scaled = preference_optimization_loss(rewards * 1000, log_probs, alpha=0.03)

        torch.testing.assert_close(original, scaled)


if __name__ == "__main__":
    unittest.main()
