"""Manuscript-exact PureMCTS and MixedMCTS expansion proposals."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List

import numpy as np

from evoflow_core import AA_ALPHABET, clean_sequence


@dataclass(frozen=True)
class MutationProposal:
    sequence: str
    positions: tuple[int, ...]
    old_residues: str
    new_residues: str
    proposal_type: str
    attempt_index: int = -1

    @property
    def hamming_distance(self) -> int:
        return len(self.positions)


def local_mutation(
    sequence: str,
    rng: np.random.Generator,
    proposal_type: str = "local",
    one_mutation_probability: float = 0.8,
    two_mutation_probability: float = 0.2,
) -> MutationProposal:
    """Draw a same-length one/two-residue canonical substitution."""

    sequence = clean_sequence(sequence)
    if (
        not 0.0 <= one_mutation_probability <= 1.0
        or not 0.0 <= two_mutation_probability <= 1.0
        or not np.isclose(one_mutation_probability + two_mutation_probability, 1.0)
    ):
        raise ValueError("One/two mutation probabilities must be in [0,1] and sum to 1")
    if len(sequence) < 2:
        raise ValueError("Manuscript local proposals require sequence length >= 2")
    count = 1 if rng.random() < one_mutation_probability else 2
    positions = tuple(sorted(int(value) for value in rng.choice(len(sequence), count, replace=False)))
    residues = list(sequence)
    old: List[str] = []
    new: List[str] = []
    for position in positions:
        old_residue = residues[position]
        choices = [value for value in AA_ALPHABET if value != old_residue]
        new_residue = str(rng.choice(choices))
        residues[position] = new_residue
        old.append(old_residue)
        new.append(new_residue)
    mutated = "".join(residues)
    if mutated == sequence:
        raise AssertionError("Local mutation unexpectedly produced the parent sequence")
    return MutationProposal(
        sequence=mutated,
        positions=positions,
        old_residues="".join(old),
        new_residues="".join(new),
        proposal_type=proposal_type,
    )


def independent_same_length_proposal(
    sequence: str,
    rng: np.random.Generator,
) -> MutationProposal:
    """Draw every residue independently from the canonical alphabet."""

    sequence = clean_sequence(sequence)
    sampled = "".join(str(value) for value in rng.choice(list(AA_ALPHABET), len(sequence)))
    positions = tuple(
        index for index, (old, new) in enumerate(zip(sequence, sampled)) if old != new
    )
    return MutationProposal(
        sequence=sampled,
        positions=positions,
        old_residues="".join(sequence[index] for index in positions),
        new_residues="".join(sampled[index] for index in positions),
        proposal_type="global_same_length",
    )


def candidate_attempt_count(
    sequence_length: int,
    max_candidates: int = 150,
    candidates_per_residue: int = 5,
) -> int:
    """Return the frozen number of proposal events for one expansion."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    return min(int(max_candidates), int(candidates_per_residue) * int(sequence_length))


def generate_proposals(
    sequence: str,
    strategy: str,
    count: int,
    rng: np.random.Generator,
    one_mutation_probability: float = 0.8,
    two_mutation_probability: float = 0.2,
    pure_local_probability: float = 1.0,
    mixed_local_probability: float = 0.5,
    mixed_global_same_length_probability: float = 0.5,
) -> list[MutationProposal]:
    """Generate exactly ``count`` ordered proposal events without resampling."""

    if strategy not in {"PureMCTS", "MixedMCTS"}:
        raise ValueError("Unknown strategy: {}".format(strategy))
    if count <= 0:
        raise ValueError("Proposal count must be positive")
    if pure_local_probability != 1.0:
        raise ValueError("PureMCTS pure_local_probability must equal 1.0")
    if (
        not 0.0 <= mixed_local_probability <= 1.0
        or not 0.0 <= mixed_global_same_length_probability <= 1.0
        or not np.isclose(
            mixed_local_probability + mixed_global_same_length_probability, 1.0
        )
    ):
        raise ValueError("Mixed proposal probabilities must be in [0,1] and sum to 1")
    proposals: list[MutationProposal] = []
    for attempt_index in range(count):
        if strategy == "PureMCTS" or rng.random() < mixed_local_probability:
            proposal = local_mutation(
                sequence,
                rng,
                proposal_type="local",
                one_mutation_probability=one_mutation_probability,
                two_mutation_probability=two_mutation_probability,
            )
        else:
            proposal = independent_same_length_proposal(sequence, rng)
        proposals.append(replace(proposal, attempt_index=attempt_index))
    return proposals
