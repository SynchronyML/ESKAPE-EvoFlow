"""Load and validate the frozen manuscript MCTS configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .io import object_sha256


MANUSCRIPT_VALUES = {
    "branching_factor": 8,
    "exploration_coefficient": 25.0,
    "uct_epsilon": 1e-6,
    "fingerprint_radius": 3,
    "fingerprint_bits": 1024,
    "max_expansions": 1000,
    "patience": 150,
    "max_candidates": 150,
    "candidates_per_residue": 5,
    "one_mutation_probability": 0.8,
    "two_mutation_probability": 0.2,
    "pure_local_probability": 1.0,
    "mixed_local_probability": 0.5,
    "mixed_global_same_length_probability": 0.5,
    "canonical_amino_acids": "ACDEFGHIKLMNPQRSTVWY",
    "preserve_length": True,
    "amp_weight": 0.4,
    "mic_weight": 0.6,
    "mic_activity_transform": "exp(-log10_mic)",
}


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("MCTS configuration must be a YAML mapping")
    return value


def flatten_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    algorithm = config["algorithm"]
    proposals = config["proposals"]
    scoring = config["scoring"]
    return {
        "branching_factor": algorithm["branching_factor"],
        "exploration_coefficient": algorithm["exploration_coefficient"],
        "uct_epsilon": algorithm["uct_epsilon"],
        "fingerprint_radius": algorithm["fingerprint_radius"],
        "fingerprint_bits": algorithm["fingerprint_bits"],
        "max_expansions": algorithm["max_expansions"],
        "patience": algorithm["patience"],
        "max_candidates": algorithm["max_candidates"],
        "candidates_per_residue": algorithm["candidates_per_residue"],
        "one_mutation_probability": proposals["one_mutation_probability"],
        "two_mutation_probability": proposals["two_mutation_probability"],
        "pure_local_probability": proposals["pure_local_probability"],
        "mixed_local_probability": proposals["mixed_local_probability"],
        "mixed_global_same_length_probability": proposals[
            "mixed_global_same_length_probability"
        ],
        "canonical_amino_acids": proposals["canonical_amino_acids"],
        "preserve_length": proposals["preserve_length"],
        "amp_weight": scoring["amp_weight"],
        "mic_weight": scoring["mic_weight"],
        "mic_activity_transform": scoring["mic_activity_transform"],
    }


def validate_manuscript_config(config: Dict[str, Any]) -> None:
    if config.get("contract_version") != "ESKAPE-EvoFlow-UCT-MCTS-MANUSCRIPT-EXACT-2026-08-29":
        raise ValueError("Unknown or missing manuscript contract_version")
    observed = flatten_contract(config)
    mismatches = {
        key: (MANUSCRIPT_VALUES[key], observed.get(key))
        for key in MANUSCRIPT_VALUES
        if observed.get(key) != MANUSCRIPT_VALUES[key]
    }
    if mismatches:
        raise ValueError("Configuration violates the manuscript contract: {}".format(mismatches))
    prohibited = {
        "algorithm.improvement_tolerance": config["algorithm"].get(
            "improvement_tolerance"
        ),
        "algorithm.max_proposal_retries": config["algorithm"].get(
            "max_proposal_retries"
        ),
        "proposals.mixed_random_local_probability": config["proposals"].get(
            "mixed_random_local_probability"
        ),
        "proposals.mixed_pure_local_probability": config["proposals"].get(
            "mixed_pure_local_probability"
        ),
    }
    present = {key: value for key, value in prohibited.items() if value is not None}
    if present:
        raise ValueError("Stale non-contract configuration keys are prohibited: {}".format(present))
    if config["scoring"].get("amp_inference_activation") != "SiLU":
        raise ValueError("MCTS AMP inference activation must be SiLU")


def config_hash(config: Dict[str, Any]) -> str:
    return object_sha256(config)
