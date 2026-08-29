import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mcts.fitness import Score
from mcts.proposals import MutationProposal
from mcts.search import MCTSSearch, derive_tree_seed, is_strict_improvement
from tests.helpers import DeterministicMockScorer, small_config


class SearchInvariantTests(unittest.TestCase):
    def test_branching_and_one_child_per_expansion(self):
        with tempfile.TemporaryDirectory() as temporary:
            scorer = DeterministicMockScorer()
            search = MCTSSearch(
                "parent",
                "ACDEFGHIKLMN",
                "PureMCTS",
                scorer,
                small_config(max_expansions=20),
                derive_tree_seed(42, "parent", "PureMCTS"),
                Path(temporary),
            )
            search.run()
            self.assertEqual(len(search.nodes), search.completed_expansions + 1)
            self.assertTrue(all(len(node.children) <= 8 for node in search.nodes.values()))
            selected = [row for row in search.candidate_rows if row["selected_as_child"]]
            self.assertEqual(len(selected), search.completed_expansions)
            self.assertEqual(
                [row["selected_node"] for row in search.trajectory_rows[:8]],
                [search.root_id] * 8,
            )
            self.assertFalse(any(row["iteration"] < 8 for row in search.selection_rows))
            self.assertNotEqual(search.trajectory_rows[8]["selected_node"], search.root_id)

    def test_batch_duplicates_score_and_insert_once(self):
        duplicates = [
            MutationProposal("CCDEFGHIK", (0,), "A", "C", "local", index)
            for index in range(27)
        ]
        with tempfile.TemporaryDirectory() as temporary, patch(
            "mcts.search.generate_proposals", return_value=duplicates
        ):
            scorer = DeterministicMockScorer()
            search = MCTSSearch(
                "parent",
                "ACDEFGHIK",
                "PureMCTS",
                scorer,
                small_config(max_expansions=1),
                123,
                Path(temporary),
            )
            root_scores = scorer.new_score_count
            search.run()
            self.assertEqual(len(search.nodes), 2)
            self.assertEqual(scorer.new_score_count - root_scores, 1)
            self.assertEqual(
                sum(row["selected_as_child"] for row in search.candidate_rows), 1
            )
            self.assertEqual(
                sum(row["eligible"] for row in search.candidate_rows), 27
            )
            self.assertEqual(len(search.candidate_rows), 27)

    def test_first_maximum_wins_without_lexicographic_tie_break(self):
        first = MutationProposal("DCDEFGHIK", (0,), "A", "D", "local", 0)
        second = MutationProposal("CCDEFGHIK", (0,), "A", "C", "local", 1)

        def equal_scores(sequences):
            return {
                sequence: Score.from_values(
                    sequence,
                    0.5,
                    [0.0] * 6,
                    embedding=np.zeros(4, dtype=np.float32),
                )
                for sequence in dict.fromkeys(sequences)
            }

        with tempfile.TemporaryDirectory() as temporary:
            search = MCTSSearch(
                "parent",
                "ACDEFGHIK",
                "PureMCTS",
                DeterministicMockScorer(),
                replace(
                    small_config(max_expansions=1),
                    max_candidates=2,
                    candidates_per_residue=1,
                ),
                123,
                Path(temporary),
            )
            with patch(
                "mcts.search.generate_proposals", return_value=[first, second]
            ), patch.object(search, "_score_map", side_effect=equal_scores):
                search.run()
            self.assertEqual(search.nodes[1].sequence, first.sequence)
            selected = [row for row in search.candidate_rows if row["selected_as_child"]]
            self.assertEqual(selected[0]["attempt_index"], 0)

    def test_strict_improvement_has_no_tolerance(self):
        best = 1.0
        self.assertFalse(is_strict_improvement(best, best))
        self.assertTrue(is_strict_improvement(np.nextafter(best, np.inf), best))


if __name__ == "__main__":
    unittest.main()
