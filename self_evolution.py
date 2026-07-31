#!/usr/bin/env python3
"""Per-seed reward-guided tree search using local peptide substitutions."""

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from evoflow_core import (
    AA_ALPHABET,
    BACTERIA,
    ESMC_WEIGHTS_HELP,
    ESMCEncoder,
    load_amp_classifier,
    load_mic_models,
    load_sequence_records,
    predict_amp,
    predict_mics,
    resolve_device,
    safe_filename,
)


@dataclass
class Node:
    sequence: str
    reward: float
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0

    def mean_reward(self) -> float:
        return self.total_reward / (self.visits + 1e-8)


class Oracle:
    def __init__(self, args):
        self.encoder = ESMCEncoder(args.esmc_weights, args.device)
        self.device = resolve_device(args.device)
        self.classifier = load_amp_classifier(
            args.weight_dir / "amp_classifier.pt", self.device
        )
        self.mic_models = load_mic_models(args.weight_dir)
        self.batch_size = args.batch_size

    def evaluate(self, sequences):
        features = self.encoder.encode(sequences, batch_size=self.batch_size)
        amp = predict_amp(self.classifier, features, self.device)
        mics = predict_mics(self.mic_models, features)
        mic_proxy = np.exp(-mics).mean(axis=1)
        rewards = 0.4 * amp + 0.6 * mic_proxy
        return rewards, amp, mics


def mutate(sequence: str, rng: np.random.Generator) -> str:
    maximum = max(1, len(sequence) // 5)
    count = int(rng.integers(1, maximum + 1))
    positions = rng.choice(len(sequence), size=count, replace=False)
    residues = list(sequence)
    for position in positions:
        choices = [amino_acid for amino_acid in AA_ALPHABET if amino_acid != residues[position]]
        residues[position] = str(rng.choice(choices))
    return "".join(residues)


def select_expansion_node(root: Node, max_children: int, exploration: float) -> Node:
    node = root
    while len(node.children) >= max_children:
        log_parent = math.log(node.visits + 1.0)
        scores = [
            child.mean_reward()
            + exploration * math.sqrt(log_parent / (child.visits + 1e-8))
            for child in node.children
        ]
        node = node.children[int(np.argmax(scores))]
    return node


def optimize_seed(identifier, seed_sequence, oracle, args, seed_offset):
    rng = np.random.default_rng(args.seed + seed_offset)
    root_reward, root_amp, root_mics = oracle.evaluate([seed_sequence])
    root = Node(
        sequence=seed_sequence,
        reward=float(root_reward[0]),
        visits=1,
        total_reward=float(root_reward[0]),
    )
    explored = {seed_sequence}
    best = root
    no_improvement = 0
    history = []

    for iteration in range(args.iterations):
        parent = select_expansion_node(root, args.max_children, args.exploration)
        attempts = min(args.max_candidates, args.candidates_per_residue * len(parent.sequence))
        candidates = []
        candidate_set = set()
        for _ in range(attempts):
            candidate = mutate(parent.sequence, rng)
            if candidate not in explored and candidate not in candidate_set:
                candidates.append(candidate)
                candidate_set.add(candidate)
        if not candidates:
            break

        rewards, amp_probabilities, mics = oracle.evaluate(candidates)
        best_index = int(np.argmax(rewards))
        sequence = candidates[best_index]
        child = Node(sequence, float(rewards[best_index]), parent=parent)
        parent.children.append(child)
        explored.add(sequence)

        current = child
        while current is not None:
            current.visits += 1
            current.total_reward += child.reward
            current = current.parent

        distance = sum(left != right for left, right in zip(parent.sequence, sequence))
        row = {
            "Iteration": iteration,
            "Seed_ID": identifier,
            "Sequence": sequence,
            "Parent_sequence": parent.sequence,
            "Mutation_count": distance,
            "Reward": child.reward,
            "AMP_probability": float(amp_probabilities[best_index]),
        }
        for column, bacterium in enumerate(BACTERIA):
            row["{}_log10_MIC".format(bacterium)] = float(mics[best_index, column])
        history.append(row)

        if child.reward > best.reward:
            best = child
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= args.patience:
            break

    history_frame = pd.DataFrame(history)
    history_path = args.output_dir / "{}_history.csv".format(safe_filename(identifier))
    history_frame.to_csv(history_path, index=False)
    return {
        "Seed_ID": identifier,
        "Seed_sequence": seed_sequence,
        "Initial_reward": float(root_reward[0]),
        "Initial_AMP_probability": float(root_amp[0]),
        "Best_sequence": best.sequence,
        "Best_reward": best.reward,
        "Expansions": len(history),
        "History_file": str(history_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run local-mutation peptide self-evolution")
    parser.add_argument("--input", type=Path, help="Seed CSV, TSV, FASTA or text file")
    parser.add_argument("--sequence", nargs="*", help="Seed sequences supplied directly")
    parser.add_argument("--sequence-column", default="Sequence")
    parser.add_argument("--output-dir", type=Path, default=Path("self_evolution_results"))
    parser.add_argument("--weight-dir", type=Path, default=Path("weight"))
    parser.add_argument(
        "--esmc-weights",
        type=Path,
        default=None,
        help=ESMC_WEIGHTS_HELP,
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--max-children", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=150)
    parser.add_argument("--candidates-per-residue", type=int, default=5)
    parser.add_argument("--exploration", type=float, default=1.41421356237)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_sequence_records(args.input, args.sequence, args.sequence_column)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oracle = Oracle(args)
    summaries = [
        optimize_seed(identifier, sequence, oracle, args, index)
        for index, (identifier, sequence) in enumerate(records)
    ]
    output = args.output_dir / "self_evolution_summary.csv"
    pd.DataFrame(summaries).to_csv(output, index=False)
    print("Saved {} self-evolution summaries to {}".format(len(summaries), output))


if __name__ == "__main__":
    main()
