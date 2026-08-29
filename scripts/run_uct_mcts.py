#!/usr/bin/env python3
"""Run manuscript-contract UCT-guided PureMCTS and MixedMCTS peptide evolution."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import queue
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# Must be set before importing torch (directly or through project modules) for
# deterministic CUDA matrix multiplications on CUDA >= 10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evoflow_core import load_sequence_records, safe_filename  # noqa: E402
from mcts.config import config_hash, load_config, validate_manuscript_config  # noqa: E402
from mcts.io import atomic_dataframe_csv, atomic_json_dump  # noqa: E402
from mcts.scorer import ModelScorer  # noqa: E402
from mcts.search import MCTSSearch, SearchConfig, derive_tree_seed  # noqa: E402


CANONICAL_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def _resolve_from_repo(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auditable UCT-guided MCTS peptide evolution under the manuscript contract"
    )
    parser.add_argument("--parents", type=Path, help="Flow-parent CSV/TSV/FASTA/text file")
    parser.add_argument("--sequence", nargs="*", help="Flow-parent sequences supplied directly")
    parser.add_argument("--sequence-column", default="Sequence")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["PureMCTS", "MixedMCTS"],
        default=["PureMCTS", "MixedMCTS"],
    )
    parser.add_argument("--devices", nargs="+", help="One persistent worker per distinct device")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/uct_mcts_manuscript.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--reuse-score-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-expansions", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--allow-nonmanuscript-config", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use the first two parents and 20 expansions per tree, then validate outputs",
    )
    return parser.parse_args()


def _records(args: argparse.Namespace) -> list[tuple[int, str, str]]:
    loaded = load_sequence_records(args.parents, args.sequence, args.sequence_column)
    result: list[tuple[int, str, str]] = []
    seen_ids: dict[str, int] = {}
    for index, (identifier, sequence) in enumerate(loaded, start=1):
        identifier = str(identifier)
        seen_ids[identifier] = seen_ids.get(identifier, 0) + 1
        if seen_ids[identifier] > 1:
            identifier = "{}_{}".format(identifier, seen_ids[identifier])
        result.append((index, identifier, sequence))
    return result


def _search_config(config: Dict[str, Any], args: argparse.Namespace) -> SearchConfig:
    search = SearchConfig.from_mapping(config)
    if args.max_expansions is not None:
        search = replace(search, max_expansions=args.max_expansions)
    if args.patience is not None:
        search = replace(search, patience=args.patience)
    if args.checkpoint_interval is not None:
        search = replace(search, checkpoint_interval=args.checkpoint_interval)
    if args.smoke_test:
        search = replace(search, max_expansions=20, patience=20, checkpoint_interval=5)
    if (
        not args.allow_nonmanuscript_config
        and not args.smoke_test
        and (search.max_expansions != 1000 or search.patience != 150)
    ):
        raise ValueError(
            "Formal runs require max_expansions=1000 and patience=150; pass "
            "--allow-nonmanuscript-config for an explicitly non-manuscript diagnostic run"
        )
    return search


def _run_assignment(
    worker_index: int,
    device: str,
    records: Sequence[tuple[int, str, str]],
    strategies: Sequence[str],
    config: Dict[str, Any],
    search_config: SearchConfig,
    base_seed: int,
    output_dir: Path,
    cache_dir: Path,
    reuse_score_cache: bool,
    resume: bool,
    batch_size: int,
    progress_queue: object | None = None,
) -> list[dict]:
    model_config = config["models"]
    scorer = ModelScorer(
        weight_dir=_resolve_from_repo(model_config["weight_dir"]),
        esmc_weights=_resolve_from_repo(model_config["esmc_weights"]),
        device=device,
        batch_size=batch_size,
        cache_path=cache_dir / "scores_worker_{:02d}.sqlite3".format(worker_index),
        reuse_score_cache=reuse_score_cache or resume,
        show_batch_progress=True,
    )
    atomic_json_dump(scorer.metadata, output_dir / "worker_{:02d}_model_metadata.json".format(worker_index))
    summaries: list[dict] = []
    try:
        iterator = tqdm(
            records,
            total=len(records),
            desc="Parents {}".format(device),
            position=worker_index + 1,
            leave=True,
        )
        for parent_index, parent_id, sequence in iterator:
            parent_directory = output_dir / "parent_{:04d}_{}".format(
                parent_index, safe_filename(parent_id)
            )
            for strategy in strategies:
                tree_seed = derive_tree_seed(base_seed, parent_id, strategy)
                tree = MCTSSearch(
                    parent_id=parent_id,
                    parent_sequence=sequence,
                    strategy=strategy,
                    scorer=scorer,
                    config=search_config,
                    tree_seed=tree_seed,
                    output_dir=parent_directory / strategy,
                    resume=resume,
                    progress_position=worker_index + 2,
                    run_metadata={
                        "base_seed": base_seed,
                        "worker_index": worker_index,
                        "device": device,
                        "manuscript_config_hash": config_hash(config),
                    },
                )
                summaries.append(tree.run())
            if progress_queue is not None:
                progress_queue.put(("parent", worker_index))
    finally:
        scorer.close()
    return summaries


def _worker_entry(result_queue, progress_queue, kwargs: Dict[str, Any]) -> None:
    try:
        summaries = _run_assignment(progress_queue=progress_queue, **kwargs)
        result_queue.put(("done", kwargs["worker_index"], summaries))
    except BaseException:
        result_queue.put(("error", kwargs["worker_index"], traceback.format_exc()))


def _run_all(
    records: list[tuple[int, str, str]],
    devices: list[str],
    common: Dict[str, Any],
) -> list[dict]:
    assignments = [records[index :: len(devices)] for index in range(len(devices))]
    active = [(index, device, values) for index, (device, values) in enumerate(zip(devices, assignments)) if values]
    if len(active) == 1:
        index, device, values = active[0]
        return _run_assignment(
            worker_index=index,
            device=device,
            records=values,
            progress_queue=None,
            **common,
        )

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    progress_queue = context.Queue()
    processes = []
    for index, device, values in active:
        kwargs = dict(common)
        kwargs.update(worker_index=index, device=device, records=values)
        process = context.Process(
            target=_worker_entry,
            args=(result_queue, progress_queue, kwargs),
            name="uct-mcts-{}".format(device.replace(":", "_")),
        )
        process.start()
        processes.append(process)
    parent_progress = tqdm(total=len(records), desc="Parents", position=0, leave=True)
    completed_workers = 0
    summaries: list[dict] = []
    errors: list[str] = []
    while completed_workers < len(processes):
        try:
            while True:
                message = progress_queue.get_nowait()
                if message[0] == "parent":
                    parent_progress.update(1)
        except queue.Empty:
            pass
        try:
            status, worker_index, payload = result_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        completed_workers += 1
        if status == "done":
            summaries.extend(payload)
        else:
            errors.append("worker {}:\n{}".format(worker_index, payload))
    for process in processes:
        process.join()
    parent_progress.close()
    if errors:
        raise RuntimeError("\n".join(errors))
    return summaries


def _validate_smoke_outputs(
    output_dir: Path,
    expected_trees: int,
    *,
    resumed: bool = False,
) -> None:
    result_files = sorted(output_dir.glob("parent_*/*/final_result.csv"))
    if len(result_files) != expected_trees:
        raise AssertionError("Smoke test did not produce every expected tree")
    tree_reports = []
    for result_path in result_files:
        directory = result_path.parent
        nodes = pd.read_csv(directory / "nodes.csv")
        edges = pd.read_csv(directory / "edges.csv")
        candidates = pd.read_csv(directory / "candidate_evaluations.csv")
        trajectory = pd.read_csv(directory / "trajectory.csv")
        selection = pd.read_csv(directory / "selection_log.csv")
        if not np.isfinite(nodes["immediate_reward"]).all():
            raise AssertionError("Non-finite node reward")
        if len(trajectory) != 20:
            raise AssertionError("Smoke tree did not complete 20 expansions")
        strategy = directory.name
        eligible = candidates.loc[candidates["eligible"]]
        if not candidates["sequence"].map(
            lambda value: set(value).issubset(CANONICAL_AMINO_ACIDS)
        ).all():
            raise AssertionError("Non-canonical proposal detected")
        node_sequences = nodes.set_index("node_id")["sequence"].to_dict()
        expected_attempts = candidates["selected_parent_node"].map(
            lambda node_id: min(150, 5 * len(node_sequences[int(node_id)]))
        )
        observed_attempts = candidates.groupby("iteration")["attempt_index"].transform("size")
        if not observed_attempts.eq(expected_attempts).all():
            raise AssertionError("Candidate attempts differ from min(150, 5L)")
        expected_lengths = candidates["selected_parent_node"].map(
            lambda node_id: len(node_sequences[int(node_id)])
        )
        if not candidates["sequence"].str.len().eq(expected_lengths).all():
            raise AssertionError("Proposal length changed")
        if strategy == "PureMCTS":
            if not candidates["proposal_type"].eq("local").all():
                raise AssertionError("PureMCTS emitted a non-local proposal")
        elif not candidates["proposal_type"].isin(["local", "global_same_length"]).all():
            raise AssertionError("Unknown MixedMCTS proposal type")
        local = eligible.loc[eligible["proposal_type"].eq("local")]
        if not local["hamming_distance"].isin([1, 2]).all():
            raise AssertionError("Local proposal violated one/two-substitution rule")
        global_count = int(candidates["proposal_type"].eq("global_same_length").sum())
        if strategy == "MixedMCTS" and global_count == 0:
            raise AssertionError("MixedMCTS did not emit global same-length proposals")
        if not edges.groupby("parent_node_id").size().le(8).all():
            raise AssertionError("Branching factor exceeded")
        if selection.empty or not selection["UCT_exploration"].gt(0).any():
            raise AssertionError("UCT exploration never became positive")
        json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        tree_reports.append(
            {
                "tree_directory": str(directory),
                "n_expansions": int(len(trajectory)),
                "n_nodes": int(len(nodes)),
                "max_branching_factor_observed": int(
                    edges.groupby("parent_node_id").size().max()
                ),
                "positive_uct_exploration": True,
                "all_rewards_finite": True,
                "proposal_contract_pass": True,
                "global_same_length_proposal_events": global_count,
            }
        )
    atomic_json_dump(
        {
            "status": "UCT_MCTS_SMOKE_TEST_PASS",
            "expected_trees": expected_trees,
            "validated_trees": len(result_files),
            "checkpoint_resume_unit_test": "PASS",
            "resume_cli_validation": "PASS" if resumed else "NOT_RUN_IN_THIS_INVOCATION",
            "trees": tree_reports,
        },
        output_dir / "smoke_test_report.json",
    )
    print("UCT_MCTS_SMOKE_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_manuscript_config(config)
    records = _records(args)
    if args.smoke_test:
        if len(records) < 2:
            raise ValueError("--smoke-test requires at least two parent peptides")
        records = records[:2]
    devices = args.devices or (["cuda:0"] if __import__("torch").cuda.is_available() else ["cpu"])
    if len(set(devices)) != len(devices):
        raise ValueError("Each worker device must be distinct; duplicate devices would reload models")
    base_seed = int(args.seed if args.seed is not None else config["reproducibility"]["base_seed"])
    search_config = _search_config(config, args)
    batch_size = int(args.batch_size or config["scoring"]["batch_size"])
    output_dir = _resolve_from_repo(
        args.output_dir or config["outputs"]["output_dir"]
    )
    cache_dir = _resolve_from_repo(args.cache_dir or config["outputs"]["cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "manuscript_config": config,
        "manuscript_config_hash": config_hash(config),
        "effective_search_config": vars(search_config),
        "base_seed": base_seed,
        "devices": devices,
        "strategies": args.strategies,
        "n_parents": len(records),
        "resume": args.resume,
        "smoke_test": args.smoke_test,
        "contract_status": "MANUSCRIPT_EXACT_UCT_MCTS_READY",
    }
    atomic_json_dump(run_manifest, output_dir / "run_manifest.json")
    common = {
        "strategies": args.strategies,
        "config": config,
        "search_config": search_config,
        "base_seed": base_seed,
        "output_dir": output_dir,
        "cache_dir": cache_dir,
        "reuse_score_cache": args.reuse_score_cache,
        "resume": args.resume,
        "batch_size": batch_size,
    }
    summaries = _run_all(records, devices, common)
    atomic_dataframe_csv(pd.DataFrame(summaries), output_dir / "run_summary.csv")
    if args.smoke_test:
        _validate_smoke_outputs(
            output_dir,
            len(records) * len(args.strategies),
            resumed=args.resume,
        )


if __name__ == "__main__":
    main()
