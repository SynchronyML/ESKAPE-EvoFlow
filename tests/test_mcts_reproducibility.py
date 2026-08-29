import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mcts.search import MCTSSearch, derive_tree_seed
from tests.helpers import DeterministicMockScorer, small_config


def run_tree(directory, strategy="MixedMCTS", resume=False, stop_after=None, scorer=None):
    scorer = scorer or DeterministicMockScorer()
    search = MCTSSearch(
        "parent-01",
        "KSYKFECRWRFHLTTNCIKT",
        strategy,
        scorer,
        small_config(max_expansions=12),
        derive_tree_seed(42, "parent-01", strategy),
        Path(directory),
        resume=resume,
    )
    search.run(stop_after=stop_after)
    return search


class ReproducibilityTests(unittest.TestCase):
    def test_same_seed_same_trajectory_and_winner(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            left = run_tree(first)
            right = run_tree(second)
            self.assertEqual(left.trajectory_rows, right.trajectory_rows)
            pd.testing.assert_frame_equal(
                pd.DataFrame(left.candidate_rows), pd.DataFrame(right.candidate_rows)
            )
            self.assertEqual(left.summary(), right.summary())

    def test_resume_matches_continuous_run(self):
        with tempfile.TemporaryDirectory() as continuous_dir, tempfile.TemporaryDirectory() as resume_dir:
            continuous = run_tree(continuous_dir)
            resume_scorer = DeterministicMockScorer()
            paused = run_tree(resume_dir, stop_after=5, scorer=resume_scorer)
            self.assertEqual(paused.completed_expansions, 5)
            resumed = run_tree(resume_dir, resume=True, scorer=resume_scorer)
            self.assertEqual(continuous.trajectory_rows, resumed.trajectory_rows)
            pd.testing.assert_frame_equal(
                pd.DataFrame(continuous.candidate_rows),
                pd.DataFrame(resumed.candidate_rows),
            )
            self.assertEqual(continuous.summary(), resumed.summary())


if __name__ == "__main__":
    unittest.main()
