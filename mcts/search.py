"""Auditable UCT-guided MCTS search and deterministic checkpoint/resume."""

from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from evoflow_core import AA_ALPHABET, clean_sequence

from .fitness import MIC_COLUMNS, SPECIES, Score
from .io import atomic_dataframe_csv, atomic_json_dump, atomic_pickle_dump, object_sha256, pickle_load
from .node import MCTSNode
from .proposals import MutationProposal, candidate_attempt_count, generate_proposals
from .uct import FingerprintIndex, select_expansion_node


@dataclass(frozen=True)
class SearchConfig:
    branching_factor: int = 8
    exploration_coefficient: float = 25.0
    uct_epsilon: float = 1e-6
    fingerprint_radius: int = 3
    fingerprint_bits: int = 1024
    max_expansions: int = 1000
    patience: int = 150
    max_candidates: int = 150
    candidates_per_residue: int = 5
    one_mutation_probability: float = 0.8
    two_mutation_probability: float = 0.2
    pure_local_probability: float = 1.0
    mixed_local_probability: float = 0.5
    mixed_global_same_length_probability: float = 0.5
    checkpoint_interval: int = 20

    @classmethod
    def from_mapping(cls, config: Dict[str, Any]) -> "SearchConfig":
        algorithm = config["algorithm"]
        proposals = config["proposals"]
        checkpointing = config.get("checkpointing", {})
        return cls(
            branching_factor=int(algorithm["branching_factor"]),
            exploration_coefficient=float(algorithm["exploration_coefficient"]),
            uct_epsilon=float(algorithm["uct_epsilon"]),
            fingerprint_radius=int(algorithm["fingerprint_radius"]),
            fingerprint_bits=int(algorithm["fingerprint_bits"]),
            max_expansions=int(algorithm["max_expansions"]),
            patience=int(algorithm["patience"]),
            max_candidates=int(algorithm["max_candidates"]),
            candidates_per_residue=int(algorithm["candidates_per_residue"]),
            one_mutation_probability=float(proposals["one_mutation_probability"]),
            two_mutation_probability=float(proposals["two_mutation_probability"]),
            pure_local_probability=float(proposals["pure_local_probability"]),
            mixed_local_probability=float(proposals["mixed_local_probability"]),
            mixed_global_same_length_probability=float(
                proposals["mixed_global_same_length_probability"]
            ),
            checkpoint_interval=int(checkpointing.get("interval_expansions", 20)),
        )


def derive_tree_seed(base_seed: int, parent_id: str, strategy: str) -> int:
    """Stable SHA256 seed; unlike Python hash(), this is process-independent."""

    payload = "{}\0{}\0{}".format(int(base_seed), parent_id, strategy).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def backpropagate(nodes: Dict[int, MCTSNode], child_id: int, reward: float) -> None:
    """Update the new child and every ancestor, including the root."""

    current_id: Optional[int] = child_id
    while current_id is not None:
        node = nodes[current_id]
        node.visit_count += 1
        node.cumulative_reward += float(reward)
        current_id = node.parent_id


def is_strict_improvement(new_reward: float, best_reward: float) -> bool:
    """Use the manuscript's literal comparison, with no hidden tolerance."""

    return float(new_reward) > float(best_reward)


def _git_commit(directory: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=directory,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


class MCTSSearch:
    """One independent parent × strategy UCT tree."""

    STATE_VERSION = 2

    def __init__(
        self,
        parent_id: str,
        parent_sequence: str,
        strategy: str,
        scorer: object,
        config: SearchConfig,
        tree_seed: int,
        output_dir: Path,
        resume: bool = False,
        progress_position: int = 0,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if strategy not in {"PureMCTS", "MixedMCTS"}:
            raise ValueError("strategy must be PureMCTS or MixedMCTS")
        self.parent_id = str(parent_id)
        self.parent_sequence = clean_sequence(parent_sequence)
        self.strategy = strategy
        self.scorer = scorer
        self.config = config
        self.tree_seed = int(tree_seed)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "tree_checkpoint.pkl"
        self.progress_position = progress_position
        self.run_metadata = run_metadata or {}
        self.tree_id = "{}:{}".format(self.parent_id, self.strategy)
        self.config_hash = object_sha256(asdict(config))
        seed_everything(self.tree_seed)
        self.rng = np.random.default_rng(self.tree_seed)

        self.nodes: Dict[int, MCTSNode] = {}
        self.root_id = 0
        self.next_node_id = 1
        self.best_node_id = 0
        self.completed_expansions = 0
        self.patience_counter = 0
        self.stop_reason: str | None = None
        self.tree_seen_sequences: set[str] = set()
        self.candidate_rows: list[dict] = []
        self.selection_rows: list[dict] = []
        self.trajectory_rows: list[dict] = []
        self.edge_rows: list[dict] = []
        self.fingerprints = FingerprintIndex(
            radius=config.fingerprint_radius, n_bits=config.fingerprint_bits
        )

        if resume:
            self._restore_checkpoint()
        else:
            self._initialize_root()

    def _score_map(self, sequences: Sequence[str]) -> Dict[str, Score]:
        return self.scorer.get_scores(sequences)  # type: ignore[attr-defined]

    def _initialize_root(self) -> None:
        score = self._score_map([self.parent_sequence])[self.parent_sequence]
        root = MCTSNode.from_score(
            score,
            node_id=0,
            parent_id=None,
            depth=0,
            creation_iteration=-1,
            proposal_type="flow_parent_root",
        )
        root.visit_count = 1
        root.cumulative_reward = root.immediate_reward
        self.nodes[root.node_id] = root
        self.tree_seen_sequences.add(root.sequence)
        self.fingerprints.add(root.node_id, root.sequence)

    def _rng_state(self) -> Dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy_legacy": np.random.get_state(),
            "numpy_generator": self.rng.bit_generator.state,
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def _restore_rng_state(self, value: Dict[str, Any]) -> None:
        random.setstate(value["python"])
        np.random.set_state(value["numpy_legacy"])
        self.rng.bit_generator.state = value["numpy_generator"]
        torch.set_rng_state(value["torch_cpu"])
        if torch.cuda.is_available() and value.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(value["torch_cuda"])

    def _state(self) -> Dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "tree_id": self.tree_id,
            "parent_id": self.parent_id,
            "parent_sequence": self.parent_sequence,
            "strategy": self.strategy,
            "tree_seed": self.tree_seed,
            "config_hash": self.config_hash,
            "nodes": {node_id: node.to_state() for node_id, node in self.nodes.items()},
            "next_node_id": self.next_node_id,
            "best_node_id": self.best_node_id,
            "completed_expansions": self.completed_expansions,
            "patience_counter": self.patience_counter,
            "stop_reason": self.stop_reason,
            "tree_seen_sequences": sorted(self.tree_seen_sequences),
            "candidate_rows": self.candidate_rows,
            "selection_rows": self.selection_rows,
            "trajectory_rows": self.trajectory_rows,
            "edge_rows": self.edge_rows,
            "cache_counters": {
                "hits": int(getattr(self.scorer.cache, "hits", 0)),
                "misses": int(getattr(self.scorer.cache, "misses", 0)),
            },
            "rng_state": self._rng_state(),
        }

    def _save_checkpoint(self) -> None:
        atomic_pickle_dump(self._state(), self.checkpoint_path)

    def _restore_checkpoint(self) -> None:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("Resume checkpoint not found: {}".format(self.checkpoint_path))
        value = pickle_load(self.checkpoint_path)
        if not isinstance(value, dict) or value.get("state_version") != self.STATE_VERSION:
            raise RuntimeError("Unsupported MCTS checkpoint schema")
        expected = {
            "tree_id": self.tree_id,
            "parent_sequence": self.parent_sequence,
            "strategy": self.strategy,
            "tree_seed": self.tree_seed,
            "config_hash": self.config_hash,
        }
        observed = {key: value.get(key) for key in expected}
        if observed != expected:
            raise RuntimeError("Resume checkpoint identity/config mismatch: {}".format(observed))
        self.nodes = {
            int(node_id): MCTSNode.from_state(state)
            for node_id, state in value["nodes"].items()
        }
        self.next_node_id = int(value["next_node_id"])
        self.best_node_id = int(value["best_node_id"])
        self.completed_expansions = int(value["completed_expansions"])
        self.patience_counter = int(value["patience_counter"])
        self.stop_reason = value.get("stop_reason")
        self.tree_seen_sequences = set(value["tree_seen_sequences"])
        self.candidate_rows = list(value["candidate_rows"])
        self.selection_rows = list(value["selection_rows"])
        self.trajectory_rows = list(value["trajectory_rows"])
        self.edge_rows = list(value["edge_rows"])
        cache_counters = value.get("cache_counters", {})
        if hasattr(self.scorer.cache, "hits"):
            self.scorer.cache.hits = int(cache_counters.get("hits", 0))
        if hasattr(self.scorer.cache, "misses"):
            self.scorer.cache.misses = int(cache_counters.get("misses", 0))
        for node in self.nodes.values():
            self.fingerprints.add(node.node_id, node.sequence)
        self._restore_rng_state(value["rng_state"])

    def _candidate_batch(self, parent: MCTSNode, iteration: int) -> tuple[list[MutationProposal], list[dict]]:
        requested = candidate_attempt_count(
            len(parent.sequence),
            self.config.max_candidates,
            self.config.candidates_per_residue,
        )
        proposals = generate_proposals(
            parent.sequence,
            self.strategy,
            requested,
            self.rng,
            one_mutation_probability=self.config.one_mutation_probability,
            two_mutation_probability=self.config.two_mutation_probability,
            pure_local_probability=self.config.pure_local_probability,
            mixed_local_probability=self.config.mixed_local_probability,
            mixed_global_same_length_probability=(
                self.config.mixed_global_same_length_probability
            ),
        )
        eligible: list[MutationProposal] = []
        rows: list[dict] = []
        batch_seen: set[str] = set()
        for proposal in proposals:
            wrong_shape = (
                len(proposal.sequence) != len(parent.sequence)
                or any(value not in AA_ALPHABET for value in proposal.sequence)
            )
            wrong_local_distance = (
                proposal.proposal_type == "local"
                and proposal.hamming_distance not in {1, 2}
            )
            unknown_type = proposal.proposal_type not in {"local", "global_same_length"}
            invalid = wrong_shape or wrong_local_distance or unknown_type
            already_in_tree = proposal.sequence in self.tree_seen_sequences
            duplicate_within_batch = proposal.sequence in batch_seen
            batch_seen.add(proposal.sequence)
            reasons: list[str] = []
            if invalid:
                reasons.append("invalid_sequence")
            if already_in_tree:
                reasons.append("already_in_tree")
            if duplicate_within_batch:
                reasons.append("duplicate_within_batch_cached_score")
            is_eligible = not invalid and not already_in_tree
            row = {
                "iteration": iteration,
                "attempt_index": proposal.attempt_index,
                "selected_parent_node": parent.node_id,
                "sequence": proposal.sequence,
                "proposal_type": proposal.proposal_type,
                "mutation_positions": ";".join(str(value) for value in proposal.positions),
                "old_residues": proposal.old_residues,
                "new_residues": proposal.new_residues,
                "hamming_distance": proposal.hamming_distance,
                "is_duplicate": duplicate_within_batch or already_in_tree,
                "duplicate_reason": ";".join(reasons),
                "eligible": is_eligible,
                "evaluated": False,
                "amp_probability": np.nan,
                **{MIC_COLUMNS[species]: np.nan for species in SPECIES},
                "mic_activity_component": np.nan,
                "reward": np.nan,
                "selected_as_child": False,
            }
            rows.append(row)
            if is_eligible:
                eligible.append(proposal)
        if len(rows) != requested:
            raise AssertionError("Candidate attempts did not match min(150, 5L)")
        return eligible, rows

    def _expand_once(self) -> bool:
        iteration = self.completed_expansions
        parent, decision_rows = select_expansion_node(
            self.nodes,
            self.root_id,
            self.fingerprints,
            self.config.branching_factor,
            self.config.exploration_coefficient,
            self.config.uct_epsilon,
            iteration,
        )
        self.selection_rows.extend(decision_rows)
        proposals, rows = self._candidate_batch(parent, iteration)
        if not proposals:
            self.candidate_rows.extend(rows)
            self.stop_reason = "no_insertable_candidate_in_attempt_batch"
            return False
        sequences = [proposal.sequence for proposal in proposals]
        scores = self._score_map(sequences)
        for row in rows:
            if row["eligible"] and row["sequence"] in scores:
                score = scores[row["sequence"]]
                row.update(score.as_row(include_sequence=False))
                row["reward"] = score.composite_reward
                row["evaluated"] = True
        # np.argmax returns the first maximum, preserving proposal-generation
        # order exactly when immediate rewards are tied.
        winner_index = int(
            np.argmax([scores[proposal.sequence].composite_reward for proposal in proposals])
        )
        winner = proposals[winner_index]
        winner_score = scores[winner.sequence]
        for row in rows:
            if row["eligible"] and row["attempt_index"] == winner.attempt_index:
                row["selected_as_child"] = True
                break
        self.candidate_rows.extend(rows)

        child_id = self.next_node_id
        self.next_node_id += 1
        child = MCTSNode.from_score(
            winner_score,
            node_id=child_id,
            parent_id=parent.node_id,
            depth=parent.depth + 1,
            creation_iteration=iteration,
            mutation_positions=winner.positions,
            mutation_from=winner.old_residues,
            mutation_to=winner.new_residues,
            proposal_type=winner.proposal_type,
        )
        self.nodes[child_id] = child
        parent.children.append(child_id)
        if len(parent.children) > self.config.branching_factor:
            raise AssertionError("Branching factor exceeded")
        self.tree_seen_sequences.add(child.sequence)
        self.fingerprints.add(child.node_id, child.sequence)
        self.edge_rows.append(
            {
                "parent_node_id": parent.node_id,
                "child_node_id": child.node_id,
                "mutation_positions": ";".join(str(value) for value in child.mutation_positions),
                "old_residues": child.mutation_from,
                "new_residues": child.mutation_to,
                "proposal_type": child.proposal_type,
            }
        )

        # No independent stochastic rollout is performed. Expanded peptide nodes
        # are directly evaluated by the frozen multi-model fitness.
        backpropagate(self.nodes, child.node_id, child.immediate_reward)
        best = self.nodes[self.best_node_id]
        if is_strict_improvement(child.immediate_reward, best.immediate_reward):
            self.best_node_id = child.node_id
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        self.completed_expansions += 1
        current_best = self.nodes[self.best_node_id]
        self.trajectory_rows.append(
            {
                "iteration": iteration,
                "selected_node": parent.node_id,
                "new_child": child.node_id,
                "new_child_reward": child.immediate_reward,
                "new_child_amp_probability": child.amp_probability,
                **{
                    "new_child_{}".format(MIC_COLUMNS[species]): value
                    for species, value in zip(SPECIES, child.log10_mic)
                },
                "current_best_sequence": current_best.sequence,
                "current_best_reward": current_best.immediate_reward,
                "tree_depth": max(node.depth for node in self.nodes.values()),
                "n_nodes": len(self.nodes),
                "patience_counter": self.patience_counter,
                "cache_hit_rate": float(self.scorer.cache.hit_rate),  # type: ignore[attr-defined]
            }
        )
        return True

    def run(self, stop_after: int | None = None) -> Dict[str, Any]:
        target = self.config.max_expansions
        if stop_after is not None:
            target = min(target, int(stop_after))
        self.stop_reason = None
        progress = tqdm(
            total=max(0, target - self.completed_expansions),
            desc="{} {}".format(self.strategy, self.parent_id),
            position=self.progress_position,
            leave=True,
        )
        while self.completed_expansions < target:
            if not self._expand_once():
                break
            progress.update(1)
            progress.set_postfix(
                best_reward="{:.6f}".format(self.nodes[self.best_node_id].immediate_reward),
                patience=self.patience_counter,
                nodes=len(self.nodes),
                cache_hit="{:.1%}".format(float(self.scorer.cache.hit_rate)),  # type: ignore[attr-defined]
            )
            if self.completed_expansions % self.config.checkpoint_interval == 0:
                self._save_checkpoint()
            if self.patience_counter >= self.config.patience:
                self.stop_reason = "early_stopping_patience"
                break
        progress.close()
        if self.stop_reason is None:
            if stop_after is not None and self.completed_expansions >= target < self.config.max_expansions:
                self.stop_reason = "checkpoint_pause"
            elif self.completed_expansions >= self.config.max_expansions:
                self.stop_reason = "max_expansions"
            else:
                self.stop_reason = "stopped"
        self._save_checkpoint()
        self._write_outputs()
        return self.summary()

    def _lineage(self) -> list[MCTSNode]:
        lineage: list[MCTSNode] = []
        current: Optional[int] = self.best_node_id
        while current is not None:
            node = self.nodes[current]
            lineage.append(node)
            current = node.parent_id
        return list(reversed(lineage))

    def summary(self) -> Dict[str, Any]:
        root = self.nodes[self.root_id]
        best = self.nodes[self.best_node_id]
        return {
            "tree_id": self.tree_id,
            "parent_id": self.parent_id,
            "parent_sequence": self.parent_sequence,
            "strategy": self.strategy,
            "parent_reward": root.immediate_reward,
            "best_evolved_sequence": best.sequence,
            "best_reward": best.immediate_reward,
            "delta_reward": best.immediate_reward - root.immediate_reward,
            "parent_amp_probability": root.amp_probability,
            "evolved_amp_probability": best.amp_probability,
            **{
                "parent_{}".format(MIC_COLUMNS[species]): value
                for species, value in zip(SPECIES, root.log10_mic)
            },
            **{
                "evolved_{}".format(MIC_COLUMNS[species]): value
                for species, value in zip(SPECIES, best.log10_mic)
            },
            "n_expansions": self.completed_expansions,
            "stop_reason": self.stop_reason,
            "tree_seed": self.tree_seed,
        }

    def _write_outputs(self) -> None:
        node_frame = pd.DataFrame(
            [self.nodes[node_id].as_node_row(self.tree_id) for node_id in sorted(self.nodes)]
        )
        atomic_dataframe_csv(node_frame, self.output_dir / "nodes.csv")
        atomic_dataframe_csv(pd.DataFrame(self.edge_rows), self.output_dir / "edges.csv")
        atomic_dataframe_csv(
            pd.DataFrame(self.candidate_rows), self.output_dir / "candidate_evaluations.csv"
        )
        atomic_dataframe_csv(
            pd.DataFrame(self.selection_rows), self.output_dir / "selection_log.csv"
        )
        atomic_dataframe_csv(
            pd.DataFrame(self.trajectory_rows), self.output_dir / "trajectory.csv"
        )
        atomic_dataframe_csv(pd.DataFrame([self.summary()]), self.output_dir / "final_result.csv")
        lineage_rows = []
        for step, node in enumerate(self._lineage()):
            row = node.as_node_row(self.tree_id)
            row["lineage_step"] = step
            row["mutation_positions"] = ";".join(str(value) for value in node.mutation_positions)
            row["mutation_from"] = node.mutation_from
            row["mutation_to"] = node.mutation_to
            lineage_rows.append(row)
        atomic_dataframe_csv(pd.DataFrame(lineage_rows), self.output_dir / "lineage.csv")
        metadata = {
            "schema_version": 1,
            "algorithm": "UCT-guided MCTS with direct frozen-model node evaluation",
            "no_independent_stochastic_rollout": True,
            "contract_status": "MANUSCRIPT_EXACT_UCT_MCTS_READY",
            "uct_formula": "Q_child/(N_child+epsilon) + c*U(child; T\\{child})*sqrt(log(N_parent+1)/(N_child+epsilon))",
            "uniqueness": "1-max Tanimoto similarity of radius-3/1024-bit non-chiral binary Morgan fingerprint to other nodes in this tree",
            "candidate_attempts": "exactly min(150, 5L) ordered proposal events; no refill after filtering",
            "tie_break": "first maximum in proposal-generation order",
            "strict_improvement": "R_new > R_best with no tolerance",
            "search_config": asdict(self.config),
            "config_hash": self.config_hash,
            "base_run_metadata": self.run_metadata,
            "model_metadata": getattr(self.scorer, "metadata", {}),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "pytorch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "git_commit": _git_commit(Path(__file__).resolve().parents[1]),
            },
            "result": self.summary(),
        }
        atomic_json_dump(metadata, self.output_dir / "summary.json")
