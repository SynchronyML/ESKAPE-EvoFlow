import unittest

import numpy as np

from evoflow_core import AA_ALPHABET
from mcts.proposals import (
    candidate_attempt_count,
    generate_proposals,
    local_mutation,
)


class MutationTests(unittest.TestCase):
    def test_single_residue_parent_is_rejected_instead_of_changing_probabilities(self):
        with self.assertRaises(ValueError):
            local_mutation("A", np.random.default_rng(1))

    def test_ten_thousand_local_mutations(self):
        parent = "ACDEFGHIKLMNPQRSTVWY"
        rng = np.random.default_rng(42)
        distances = []
        for _ in range(10_000):
            proposal = local_mutation(parent, rng)
            distance = sum(a != b for a, b in zip(parent, proposal.sequence))
            distances.append(distance)
            self.assertEqual(len(parent), len(proposal.sequence))
            self.assertNotEqual(parent, proposal.sequence)
            self.assertTrue(set(proposal.sequence).issubset(set(AA_ALPHABET)))
            self.assertIn(distance, {1, 2})
        one_fraction = distances.count(1) / len(distances)
        self.assertTrue(0.78 <= one_fraction <= 0.82, one_fraction)

    def test_candidate_attempt_counts(self):
        self.assertEqual(
            [candidate_attempt_count(length) for length in (10, 20, 30, 50)],
            [50, 100, 150, 150],
        )

    def test_mixed_composition_and_constraints(self):
        parent = "KSYKFECRWRFHLTTNCIKT"
        proposals = generate_proposals(
            parent, "MixedMCTS", 10_000, np.random.default_rng(7)
        )
        local = [proposal for proposal in proposals if proposal.proposal_type == "local"]
        global_proposals = [
            proposal for proposal in proposals if proposal.proposal_type == "global_same_length"
        ]
        self.assertTrue(0.48 <= len(local) / len(proposals) <= 0.52)
        self.assertTrue(0.48 <= len(global_proposals) / len(proposals) <= 0.52)
        for proposal in proposals:
            distance = sum(a != b for a, b in zip(parent, proposal.sequence))
            self.assertEqual(len(parent), len(proposal.sequence))
            self.assertTrue(set(proposal.sequence).issubset(set(AA_ALPHABET)))
            if proposal.proposal_type == "local":
                self.assertIn(distance, {1, 2})
        self.assertTrue(any(proposal.hamming_distance > 2 for proposal in global_proposals))

    def test_pure_is_one_hundred_percent_local(self):
        proposals = generate_proposals(
            "KSYKFECRWRFHLTTNCIKT",
            "PureMCTS",
            1_000,
            np.random.default_rng(19),
        )
        self.assertTrue(all(proposal.proposal_type == "local" for proposal in proposals))


if __name__ == "__main__":
    unittest.main()
