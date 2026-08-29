"""MCTS node representation with serializable search statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .fitness import MIC_COLUMNS, SPECIES, Score


@dataclass
class MCTSNode:
    sequence: str
    node_id: int
    parent_id: Optional[int]
    depth: int
    immediate_reward: float
    amp_probability: float
    log10_mic: tuple[float, float, float, float, float, float]
    children: List[int] = field(default_factory=list)
    visit_count: int = 0
    cumulative_reward: float = 0.0
    creation_iteration: int = -1
    mutation_positions: tuple[int, ...] = ()
    mutation_from: str = ""
    mutation_to: str = ""
    proposal_type: str = "root"

    @property
    def mean_reward(self) -> float:
        return self.cumulative_reward / self.visit_count if self.visit_count else 0.0

    @classmethod
    def from_score(
        cls,
        score: Score,
        node_id: int,
        parent_id: Optional[int],
        depth: int,
        creation_iteration: int,
        mutation_positions: tuple[int, ...] = (),
        mutation_from: str = "",
        mutation_to: str = "",
        proposal_type: str = "root",
    ) -> "MCTSNode":
        return cls(
            sequence=score.sequence,
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            immediate_reward=score.composite_reward,
            amp_probability=score.amp_probability,
            log10_mic=score.log10_mic,
            creation_iteration=creation_iteration,
            mutation_positions=mutation_positions,
            mutation_from=mutation_from,
            mutation_to=mutation_to,
            proposal_type=proposal_type,
        )

    def as_node_row(self, tree_id: str) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "tree_id": tree_id,
            "node_id": self.node_id,
            "parent_node_id": self.parent_id,
            "sequence": self.sequence,
            "depth": self.depth,
            "visit_count": self.visit_count,
            "cumulative_reward": self.cumulative_reward,
            "mean_reward": self.mean_reward,
            "immediate_reward": self.immediate_reward,
            "amp_probability": self.amp_probability,
            "proposal_type": self.proposal_type,
            "created_iteration": self.creation_iteration,
        }
        for species, value in zip(SPECIES, self.log10_mic):
            row[MIC_COLUMNS[species]] = value
        return row

    def to_state(self) -> Dict[str, Any]:
        value = asdict(self)
        value["log10_mic"] = list(self.log10_mic)
        value["mutation_positions"] = list(self.mutation_positions)
        return value

    @classmethod
    def from_state(cls, value: Dict[str, Any]) -> "MCTSNode":
        restored = dict(value)
        restored["log10_mic"] = tuple(restored["log10_mic"])
        restored["mutation_positions"] = tuple(restored["mutation_positions"])
        return cls(**restored)
