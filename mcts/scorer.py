"""Batched ESM C, SiLU AMP and six-regressor frozen scoring pipeline."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from evoflow_core import ESMCEncoder, _find_esmc_checkpoint, _torch_load, clean_sequence, load_mic_models

from .cache import MemoryScoreCache, SQLiteScoreCache
from .fitness import MIC_COLUMNS, SPECIES, Score, composite_fitness


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class SiLUAMPInferenceMLP(nn.Module):
    """Historical generation/MCTS forward graph using the trained state dict."""

    def __init__(self, input_dim: int = 1152):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(-1)


class ModelScorer:
    """Load all frozen models once and score exact sequences in GPU batches."""

    def __init__(
        self,
        weight_dir: Path,
        esmc_weights: Path,
        device: str,
        batch_size: int,
        cache_path: Path | None = None,
        reuse_score_cache: bool = False,
        show_batch_progress: bool = True,
    ) -> None:
        self.weight_dir = Path(weight_dir).resolve()
        self.esmc_weights = Path(esmc_weights).resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable: {}".format(device))
        self.batch_size = int(batch_size)
        self.show_batch_progress = show_batch_progress
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.esmc_checkpoint = _find_esmc_checkpoint(self.esmc_weights).resolve()
        self.amp_checkpoint = (self.weight_dir / "amp_classifier.pt").resolve()
        self.mic_paths = {
            species: (self.weight_dir / "MIC_{}.joblib".format(species.replace(" ", "_"))).resolve()
            for species in SPECIES
        }
        required_paths = [self.esmc_checkpoint, self.amp_checkpoint, *self.mic_paths.values()]
        missing = [str(path) for path in required_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing model checkpoints: {}".format(missing))

        self.checkpoint_hashes = {
            "esmc": sha256_file(self.esmc_checkpoint),
            "amp": sha256_file(self.amp_checkpoint),
            **{"mic_{}".format(species.replace(" ", "_")): sha256_file(path) for species, path in self.mic_paths.items()},
        }
        cache_metadata = {
            "schema_version": 1,
            "checkpoint_hashes": self.checkpoint_hashes,
            "amp_inference_activation": "SiLU",
            "amp_dropout_probability": 0.2,
            "fitness": "0.4*pAMP+0.6*mean(exp(-log10MIC_j))",
            "species_order": list(SPECIES),
            "pooling": "project ESMCEncoder mean over non-padding tokens",
            "device": str(self.device),
            "torch_dtype": "float32",
            "runtime_versions": {
                package: importlib.metadata.version(package)
                for package in ("numpy", "torch", "scikit-learn", "joblib", "esm")
            },
        }
        self.cache = (
            SQLiteScoreCache(cache_path, cache_metadata, reuse=reuse_score_cache)
            if cache_path is not None
            else MemoryScoreCache()
        )

        self.encoder = ESMCEncoder(self.esmc_weights, str(self.device))
        self.encoder.model.eval()
        self.amp_model = SiLUAMPInferenceMLP().to(self.device)
        amp_state = _torch_load(self.amp_checkpoint, map_location=self.device)
        if isinstance(amp_state, dict) and "model" in amp_state:
            amp_state = amp_state["model"]
        incompatible = self.amp_model.load_state_dict(amp_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "AMP checkpoint mismatch: missing={}, unexpected={}".format(
                    incompatible.missing_keys, incompatible.unexpected_keys
                )
            )
        self.amp_model.eval()
        self.mic_models = load_mic_models(self.weight_dir)
        self._verify_model_state()
        self._print_startup_metadata()

    def _verify_model_state(self) -> None:
        if self.encoder.model.training or self.amp_model.training:
            raise RuntimeError("All torch models must be in eval mode")
        dropout_states = [module.training for module in self.amp_model.modules() if isinstance(module, nn.Dropout)]
        if not dropout_states or any(dropout_states):
            raise RuntimeError("MCTS AMP Dropout modules must exist and be disabled")
        if set(self.mic_models) != set(SPECIES):
            raise RuntimeError("Exactly six independent species-specific MIC models are required")

    @property
    def metadata(self) -> Dict[str, object]:
        dtype = str(next(self.encoder.model.parameters()).dtype)
        dropout = [
            {"p": float(module.p), "training": bool(module.training)}
            for module in self.amp_model.modules()
            if isinstance(module, nn.Dropout)
        ]
        return {
            "encoder_checkpoint": str(self.esmc_checkpoint),
            "amp_checkpoint": str(self.amp_checkpoint),
            "mic_checkpoints": {species: str(path) for species, path in self.mic_paths.items()},
            "checkpoint_sha256": self.checkpoint_hashes,
            "device": str(self.device),
            "dtype": dtype,
            "amp_inference_activation": "SiLU",
            "amp_dropout": dropout,
            "encoder_eval": not self.encoder.model.training,
            "amp_eval": not self.amp_model.training,
            "mic_backend": "independent joblib/scikit-learn estimators; batch CPU predict",
        }

    def _print_startup_metadata(self) -> None:
        print(json.dumps(self.metadata, indent=2, sort_keys=True, ensure_ascii=False), flush=True)

    def _ensure_scores(self, cleaned: Sequence[str]) -> Dict[str, Score]:
        unique = list(dict.fromkeys(cleaned))
        scores = self.cache.get_many(unique)
        missing = [sequence for sequence in unique if sequence not in scores]
        generated: list[Score] = []
        batches = range(0, len(missing), self.batch_size)
        progress = tqdm(
            batches,
            total=(len(missing) + self.batch_size - 1) // self.batch_size,
            desc="Scoring uncached batches",
            disable=not self.show_batch_progress or not missing,
            leave=False,
        )
        for start in progress:
            batch = missing[start : start + self.batch_size]
            embeddings = self.encoder.encode(batch, batch_size=len(batch))
            with torch.inference_mode():
                values = torch.as_tensor(embeddings, device=self.device, dtype=torch.float32)
                amp = torch.sigmoid(self.amp_model(values)).float().cpu().numpy().astype(np.float64)
            mic_columns = [
                np.asarray(self.mic_models[species].predict(embeddings), dtype=np.float64)
                for species in SPECIES
            ]
            mic = np.column_stack(mic_columns)
            activity, reward = composite_fitness(amp, mic)
            for index, sequence in enumerate(batch):
                generated.append(
                    Score(
                        sequence=sequence,
                        amp_probability=float(amp[index]),
                        log10_mic=tuple(float(value) for value in mic[index]),  # type: ignore[arg-type]
                        mic_activity_component=float(activity[index]),
                        composite_reward=float(reward[index]),
                        embedding=np.asarray(embeddings[index], dtype=np.float32),
                    )
                )
        self.cache.put_many(generated)
        scores.update({score.sequence: score for score in generated})
        return scores

    def score_sequences(self, sequences: list[str] | Sequence[str]) -> pd.DataFrame:
        """Return one deterministic score row per input sequence, preserving order.

        Every uncached sequence is newly encoded by frozen ESM C. The ESM C and
        AMP forwards are batched on the selected torch device; the six archived
        joblib regressors are independently batch-predicted on CPU.
        """

        cleaned = [clean_sequence(value) for value in sequences]
        if not cleaned:
            return pd.DataFrame(
                columns=[
                    "sequence",
                    "amp_probability",
                    *[MIC_COLUMNS[species] for species in SPECIES],
                    "mic_activity_component",
                    "composite_reward",
                ]
            )
        scores = self._ensure_scores(cleaned)
        return pd.DataFrame([scores[sequence].as_row() for sequence in cleaned])

    def get_scores(self, sequences: Sequence[str]) -> Dict[str, Score]:
        cleaned = [clean_sequence(value) for value in sequences]
        return self._ensure_scores(cleaned)

    def close(self) -> None:
        self.cache.close()
