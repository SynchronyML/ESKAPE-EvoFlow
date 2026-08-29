"""Equation-8 UCT policy with non-self molecular uniqueness."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .node import MCTSNode

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
except ImportError as exc:  # pragma: no cover - exercised as an installation failure
    raise ImportError(
        "RDKit is required for the manuscript Morgan/Tanimoto uniqueness term"
    ) from exc


@dataclass(frozen=True)
class UCTComponents:
    total: float
    exploitation: float
    exploration: float
    uniqueness: float
    parent_visits: int
    child_visits: int


class FingerprintIndex:
    def __init__(self, radius: int = 3, n_bits: int = 1024):
        self.radius = int(radius)
        self.n_bits = int(n_bits)
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            fpSize=self.n_bits,
            includeChirality=False,
        )
        self.include_chirality = False
        self._fingerprints: Dict[int, object] = {}

    def add(self, node_id: int, sequence: str) -> None:
        molecule = Chem.MolFromSequence(sequence)
        if molecule is None:
            raise ValueError("RDKit could not construct peptide molecule: {}".format(sequence))
        self._fingerprints[node_id] = self.generator.GetFingerprint(molecule)

    def uniqueness(self, node_id: int) -> float:
        """Return 1 - maximum similarity to all *other* tree nodes."""

        fingerprint = self._fingerprints[node_id]
        references = [
            value for other_id, value in self._fingerprints.items() if other_id != node_id
        ]
        if not references:
            return 1.0
        maximum = max(
            float(DataStructs.TanimotoSimilarity(fingerprint, reference))
            for reference in references
        )
        return max(0.0, min(1.0, 1.0 - maximum))


def uct_components(
    parent: MCTSNode,
    child: MCTSNode,
    fingerprints: FingerprintIndex,
    exploration_coefficient: float = 25.0,
    epsilon: float = 1e-6,
) -> UCTComponents:
    if child.visit_count <= 0:
        raise ValueError("Every inserted child must be back-propagated before selection")
    if parent.visit_count <= 0:
        raise ValueError("Parent visit_count must be positive")
    uniqueness = fingerprints.uniqueness(child.node_id)
    exploitation = child.cumulative_reward / (child.visit_count + epsilon)
    exploration = (
        exploration_coefficient
        * uniqueness
        * math.sqrt(math.log(parent.visit_count + 1.0) / (child.visit_count + epsilon))
    )
    return UCTComponents(
        total=exploitation + exploration,
        exploitation=exploitation,
        exploration=exploration,
        uniqueness=uniqueness,
        parent_visits=parent.visit_count,
        child_visits=child.visit_count,
    )


def select_expansion_node(
    nodes: Dict[int, MCTSNode],
    root_id: int,
    fingerprints: FingerprintIndex,
    branching_factor: int,
    exploration_coefficient: float,
    epsilon: float,
    iteration: int,
) -> tuple[MCTSNode, list[dict]]:
    """Descend by UCT until reaching a node with fewer than eight children."""

    node = nodes[root_id]
    decision_rows: list[dict] = []
    traversal_step = 0
    if any(len(value.children) > branching_factor for value in nodes.values()):
        raise AssertionError("Branching factor exceeded before UCT selection")
    while len(node.children) == branching_factor:
        scored: list[tuple[MCTSNode, UCTComponents]] = []
        for child_id in node.children:
            child = nodes[child_id]
            scored.append(
                (
                    child,
                    uct_components(
                        node,
                        child,
                        fingerprints,
                        exploration_coefficient,
                        epsilon,
                    ),
                )
            )
        selected, selected_components = max(
            scored, key=lambda item: (item[1].total, -item[0].node_id)
        )
        for child, components in scored:
            decision_rows.append(
                {
                    "iteration": iteration,
                    "traversal_step": traversal_step,
                    "parent_node_id": node.node_id,
                    "child_node_id": child.node_id,
                    "UCT_total": components.total,
                    "UCT_exploitation": components.exploitation,
                    "UCT_exploration": components.exploration,
                    "uniqueness_term": components.uniqueness,
                    "N_parent": components.parent_visits,
                    "N_child": components.child_visits,
                    "selected": child.node_id == selected.node_id,
                }
            )
        node = selected
        traversal_step += 1
    return node, decision_rows
