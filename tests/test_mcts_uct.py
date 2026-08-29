import math
import unittest

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from mcts.node import MCTSNode
from mcts.search import backpropagate
from mcts.uct import FingerprintIndex, select_expansion_node, uct_components


def node(node_id, parent_id, visits, cumulative, sequence="ACDEFGHIK"):
    return MCTSNode(
        sequence=sequence,
        node_id=node_id,
        parent_id=parent_id,
        depth=0 if parent_id is None else 1,
        immediate_reward=1.0,
        amp_probability=0.5,
        log10_mic=(0.0,) * 6,
        visit_count=visits,
        cumulative_reward=cumulative,
    )


class ConstantUniqueness:
    def uniqueness(self, node_id):
        return 0.5


class UCTTests(unittest.TestCase):
    def test_exact_manuscript_uct_equation(self):
        epsilon = 1e-6
        parent = node(0, None, 17, 4.0)
        child = node(1, 0, 3, 2.25)
        observed = uct_components(
            parent,
            child,
            ConstantUniqueness(),
            exploration_coefficient=25.0,
            epsilon=epsilon,
        )
        expected_exploitation = 2.25 / (3 + epsilon)
        expected_exploration = 25.0 * 0.5 * math.sqrt(
            math.log(17 + 1.0) / (3 + epsilon)
        )
        self.assertEqual(observed.exploitation, expected_exploitation)
        self.assertEqual(observed.exploration, expected_exploration)
        self.assertEqual(observed.total, expected_exploitation + expected_exploration)

    def test_lower_visit_child_has_more_exploration(self):
        parent = node(0, None, 20, 10.0)
        epsilon = 1e-6
        low_visit = node(1, 0, 2, 0.5 * (2 + epsilon))
        high_visit = node(2, 0, 8, 0.5 * (8 + epsilon))
        low = uct_components(parent, low_visit, ConstantUniqueness())
        high = uct_components(parent, high_visit, ConstantUniqueness())
        self.assertEqual(low.exploitation, high.exploitation)
        self.assertGreater(low.exploration, high.exploration)
        self.assertGreater(low.total, low.exploitation)

    def test_uniqueness_excludes_self(self):
        index = FingerprintIndex(radius=3, n_bits=1024)
        self.assertEqual(index.radius, 3)
        self.assertEqual(index.n_bits, 1024)
        self.assertFalse(index.include_chirality)
        index.add(0, "ACDEFGHIK")
        self.assertEqual(index.uniqueness(0), 1.0)
        index.add(1, "ACDEYGHIK")
        uniqueness = index.uniqueness(1)
        self.assertGreater(uniqueness, 0.0)
        self.assertLessEqual(uniqueness, 1.0)
        expected_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=3,
            fpSize=1024,
            includeChirality=False,
        )
        first = expected_generator.GetFingerprint(Chem.MolFromSequence("ACDEFGHIK"))
        second = expected_generator.GetFingerprint(Chem.MolFromSequence("ACDEYGHIK"))
        expected = 1.0 - DataStructs.TanimotoSimilarity(second, first)
        self.assertEqual(uniqueness, expected)

    def test_more_than_eight_children_is_impossible(self):
        root = node(0, None, 10, 5.0)
        nodes = {0: root}
        index = FingerprintIndex()
        index.add(0, root.sequence)
        for child_id in range(1, 10):
            child = node(child_id, 0, 1, 1.0, sequence="ACDEFGHI{}".format("K" if child_id % 2 else "L"))
            nodes[child_id] = child
            root.children.append(child_id)
            index.add(child_id, child.sequence)
        with self.assertRaises(AssertionError):
            select_expansion_node(nodes, 0, index, 8, 25.0, 1e-6, 0)

    def test_depth_three_backpropagation(self):
        nodes = {
            0: node(0, None, 1, 0.5),
            1: node(1, 0, 1, 0.6),
            2: node(2, 1, 1, 0.7),
            3: node(3, 2, 0, 0.0),
            4: node(4, 1, 9, 9.0),
        }
        before = {key: (value.visit_count, value.cumulative_reward) for key, value in nodes.items()}
        backpropagate(nodes, 3, 0.9)
        for key in (0, 1, 2, 3):
            self.assertEqual(nodes[key].visit_count, before[key][0] + 1)
            self.assertAlmostEqual(nodes[key].cumulative_reward, before[key][1] + 0.9)
        self.assertEqual(
            (nodes[4].visit_count, nodes[4].cumulative_reward), before[4]
        )


if __name__ == "__main__":
    unittest.main()
