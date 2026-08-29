import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from mcts.cache import MemoryScoreCache
from mcts.config import load_config, validate_manuscript_config
from mcts.fitness import Score
from mcts.search import MCTSSearch, SearchConfig
from tests.helpers import small_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ConstantRewardScorer:
    def __init__(self):
        self.cache = MemoryScoreCache()
        self.metadata = {"backend": "constant-reward-test-double"}

    def get_scores(self, sequences):
        scores = {}
        for sequence in dict.fromkeys(sequences):
            scores[sequence] = Score.from_values(
                sequence,
                0.5,
                [0.0] * 6,
                embedding=np.zeros(4, dtype=np.float32),
            )
        return scores


class ManuscriptContractTests(unittest.TestCase):
    def test_frozen_config_has_exact_values_and_no_tolerance_or_retry(self):
        config = load_config(REPOSITORY_ROOT / "configs/uct_mcts_manuscript.yaml")
        validate_manuscript_config(config)
        search = SearchConfig.from_mapping(config)
        self.assertEqual(search.max_expansions, 1000)
        self.assertEqual(search.patience, 150)
        self.assertEqual(search.uct_epsilon, 1e-6)
        self.assertEqual(search.mixed_local_probability, 0.5)
        self.assertEqual(search.mixed_global_same_length_probability, 0.5)
        self.assertNotIn("improvement_tolerance", config["algorithm"])
        self.assertNotIn("max_proposal_retries", config["algorithm"])

    def test_equal_rewards_increment_patience_and_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            search = MCTSSearch(
                "parent",
                "ACDEFGHIKLMN",
                "PureMCTS",
                ConstantRewardScorer(),
                replace(
                    small_config(max_expansions=20),
                    patience=3,
                    max_candidates=8,
                    candidates_per_residue=1,
                ),
                42,
                Path(temporary),
            )
            search.run()
            self.assertEqual(search.completed_expansions, 3)
            self.assertEqual(search.patience_counter, 3)
            self.assertEqual(search.stop_reason, "early_stopping_patience")
            summary = json.loads(
                (Path(temporary) / "summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(summary["no_independent_stochastic_rollout"])
            self.assertEqual(summary["tie_break"], "first maximum in proposal-generation order")
            self.assertEqual(summary["strict_improvement"], "R_new > R_best with no tolerance")


if __name__ == "__main__":
    unittest.main()
