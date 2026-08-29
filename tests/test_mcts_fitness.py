import unittest

import numpy as np

from mcts.fitness import composite_fitness


class FitnessTests(unittest.TestCase):
    def test_manuscript_formula(self):
        amp = np.array([0.75])
        mic = np.array([[0.0, 0.5, -0.5, 1.0, -1.0, 0.25]])
        activity, reward = composite_fitness(amp, mic)
        expected_activity = np.mean(np.exp(-mic[0]))
        expected_reward = 0.4 * amp[0] + 0.6 * expected_activity
        self.assertAlmostEqual(activity[0], expected_activity, places=15)
        self.assertAlmostEqual(reward[0], expected_reward, places=15)


if __name__ == "__main__":
    unittest.main()
