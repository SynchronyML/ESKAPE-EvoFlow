"""Shared inference components for the public ESKAPE-EvoFlow scripts."""

import csv
import inspect
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
from torch import nn


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"]
VOCAB = SPECIAL_TOKENS + list(AA_ALPHABET) + ["X"]
CANONICAL_TOKEN_START = len(SPECIAL_TOKENS)
CANONICAL_TOKEN_END = CANONICAL_TOKEN_START + len(AA_ALPHABET)
ESMC_MODEL_NAME = "esmc_600m"
ESMC_HIDDEN_SIZE = 1152
ESMC_HF_REPO = "biohub/esmc-600m-2024-12"
ESMC_HF_URL = "https://huggingface.co/{}".format(ESMC_HF_REPO)
ESMC_OFFLINE_EXAMPLE = "external_models/esmc-600m-2024-12"
ESMC_WEIGHTS_HELP = (
    "Optional official ESM C model directory or checkpoint for offline use. "
    "Download it from {} and place it, for example, at {}; when omitted, "
    "ESMC.from_pretrained({!r}) uses the official Hugging Face cache."
).format(ESMC_HF_URL, ESMC_OFFLINE_EXAMPLE, ESMC_MODEL_NAME)

BACTERIA = [
    "Acinetobacter baumannii",
    "Enterobacter cloacae",
    "Enterococcus faecium",
    "Klebsiella pneumoniae",
    "Pseudomonas aeruginosa",
    "Staphylococcus aureus",
]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def clean_sequence(value: str, max_length: Optional[int] = None) -> str:
    sequence = re.sub(r"\s+", "", str(value)).upper()
    if not sequence:
        raise ValueError("Encountered an empty peptide sequence")
    invalid = sorted(set(sequence) - set(AA_ALPHABET))
    if invalid:
        raise ValueError(
            "Only the 20 canonical amino acids are accepted; found: "
            + ", ".join(invalid)
        )
    if max_length is not None and len(sequence) > max_length:
        raise ValueError(
            "Sequence length {} exceeds the maximum {}".format(len(sequence), max_length)
        )
    return sequence


def load_sequence_records(
    path: Optional[Path],
    sequences: Optional[Sequence[str]] = None,
    sequence_column: str = "Sequence",
) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    if sequences:
        for index, sequence in enumerate(sequences, start=1):
            records.append(("seq{}".format(index), clean_sequence(sequence)))

    if path is None:
        if not records:
            raise ValueError("Provide --input or at least one value through --sequence")
        return records

    suffix = path.suffix.lower()
    if suffix in {".fa", ".fasta", ".faa"}:
        identifier: Optional[str] = None
        chunks: List[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if identifier is not None:
                        records.append((identifier, clean_sequence("".join(chunks))))
                    identifier = line[1:].strip() or "seq{}".format(len(records) + 1)
                    chunks = []
                else:
                    chunks.append(line)
        if identifier is not None:
            records.append((identifier, clean_sequence("".join(chunks))))
    elif suffix in {".csv", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if not reader.fieldnames:
                raise ValueError("The input table has no header")
            names = {name.lower(): name for name in reader.fieldnames}
            actual_column = names.get(sequence_column.lower())
            if actual_column is None:
                raise ValueError(
                    "Sequence column {!r} was not found; available columns: {}".format(
                        sequence_column, reader.fieldnames
                    )
                )
            id_column = names.get("id") or names.get("identifier") or names.get("name")
            for row_index, row in enumerate(reader, start=1):
                raw = row.get(actual_column)
                if raw is None or not str(raw).strip():
                    continue
                identifier = row.get(id_column) if id_column else None
                records.append(
                    (identifier or "seq{}".format(row_index), clean_sequence(raw))
                )
    elif suffix in {".txt", ".seq"}:
        with path.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle, start=1):
                line = line.strip()
                if line:
                    records.append(("seq{}".format(row_index), clean_sequence(line)))
    else:
        raise ValueError("Unsupported input format: {}".format(path.suffix))

    if not records:
        raise ValueError("No peptide sequences were found")
    return records


def _torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _find_esmc_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(
            "ESM C path does not exist: {}. Download the official model from {} "
            "and place it at {}, or pass the downloaded checkpoint directly.".format(
                path, ESMC_HF_URL, ESMC_OFFLINE_EXAMPLE
            )
        )
    candidates = sorted(list(path.glob("**/*.pth")) + list(path.glob("**/*.pt")))
    if not candidates:
        raise FileNotFoundError(
            "No .pth or .pt checkpoint found under {}. The recommended layout is "
            "{}/data/weights/esmc_600m_2024_12_v0.pth; obtain it from {}.".format(
                path, ESMC_OFFLINE_EXAMPLE, ESMC_HF_URL
            )
        )
    return next((item for item in candidates if "esmc" in item.name.lower()), candidates[0])


def load_esmc600m(path: Optional[Path], device: torch.device):
    try:
        from esm.models.esmc import ESMC
        from esm.tokenization import EsmSequenceTokenizer
    except ImportError as exc:
        raise ImportError(
            "The EvolutionaryScale esm package is required to load ESM C. "
            "Install the pinned dependency from requirements.txt, then obtain the "
            "official model from {}.".format(ESMC_HF_URL)
        ) from exc

    if path is None:
        try:
            # Loading on CPU avoids the upstream CUDA bfloat16 cast and preserves
            # the float32 representation used by the released predictors.
            model = ESMC.from_pretrained(
                ESMC_MODEL_NAME, device=torch.device("cpu")
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the official ESM C-600M model through "
                "ESMC.from_pretrained({!r}). Download the official model from {} "
                "or provide --esmc-weights {}, pointing to the downloaded model "
                "directory or checkpoint.".format(
                    ESMC_MODEL_NAME, ESMC_HF_URL, ESMC_OFFLINE_EXAMPLE
                )
            ) from exc
        model = model.to(device=device, dtype=torch.float32)
        model.eval()
        return model

    checkpoint = _find_esmc_checkpoint(path)
    config = {
        "d_model": 1152,
        "n_layers": 36,
        "n_heads": 18,
        "use_rotary_embeddings": True,
        "alphabet_size": 33,
        "pad_token_id": 1,
        "mask_token_id": 32,
    }
    for config_path in (checkpoint.parent / "config.json", path / "config.json"):
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                config.update(loaded)
            break

    tokenizer = EsmSequenceTokenizer()
    valid_keys = set(inspect.signature(ESMC.__init__).parameters)
    kwargs = {key: value for key, value in config.items() if key in valid_keys}
    kwargs.pop("self", None)
    if "tokenizer" in valid_keys:
        kwargs["tokenizer"] = tokenizer

    model = ESMC(**kwargs).to(device)
    state = _torch_load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "ESM C checkpoint mismatch. missing_keys={}, unexpected_keys={}".format(
                incompatible.missing_keys, incompatible.unexpected_keys
            )
        )
    model.eval()
    return model


class ESMCEncoder:
    """Project-compatible mean pooling over all non-padding ESM C tokens."""

    def __init__(self, weights: Optional[Path] = None, device: str = "auto"):
        self.device = resolve_device(device)
        self.model = load_esmc600m(weights, self.device)
        tokenizer = getattr(self.model, "tokenizer", None)
        self.pad_token_id = int(getattr(tokenizer, "pad_token_id", 1))

    @torch.no_grad()
    def encode(self, sequences: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not sequences:
            return np.empty((0, ESMC_HIDDEN_SIZE), dtype=np.float32)
        output: List[np.ndarray] = []
        for start in range(0, len(sequences), batch_size):
            batch = [clean_sequence(value) for value in sequences[start : start + batch_size]]
            input_ids = self.model._tokenize(batch).to(self.device)
            attention_mask = input_ids.ne(self.pad_token_id)
            result = self.model(input_ids, attention_mask)
            embeddings = result.embeddings if hasattr(result, "embeddings") else result[0]
            mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
            pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
            if pooled.shape[-1] != ESMC_HIDDEN_SIZE:
                raise RuntimeError(
                    "Expected {}-dimensional ESM C embeddings, got {}".format(
                        ESMC_HIDDEN_SIZE, pooled.shape[-1]
                    )
                )
            values = pooled.float().cpu().numpy()
            if not np.all(np.isfinite(values)) or np.any(np.linalg.norm(values, axis=1) <= 0):
                raise RuntimeError("ESM C returned non-finite or zero-norm embeddings")
            output.append(values)
        return np.vstack(output).astype(np.float32)


class AMPClassifier(nn.Module):
    """Architecture used to train the released AMP classifier checkpoint."""

    def __init__(self, input_dim: int = 1152):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(-1)


def load_amp_classifier(path: Path, device: torch.device) -> AMPClassifier:
    model = AMPClassifier().to(device)
    state = _torch_load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_amp(
    model: AMPClassifier,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    probabilities: List[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        values = torch.as_tensor(features[start : start + batch_size], device=device)
        probabilities.append(torch.sigmoid(model(values)).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float64)


def mic_model_filename(bacterium: str) -> str:
    return "MIC_{}.joblib".format(bacterium.replace(" ", "_"))


def load_mic_models(weight_dir: Path) -> Dict[str, object]:
    models: Dict[str, object] = {}
    for bacterium in BACTERIA:
        path = weight_dir / mic_model_filename(bacterium)
        if not path.exists():
            raise FileNotFoundError("Missing MIC regressor: {}".format(path))
        models[bacterium] = joblib.load(path)
    return models


def predict_mics(models: Dict[str, object], features: np.ndarray) -> np.ndarray:
    columns = [np.asarray(models[name].predict(features), dtype=float) for name in BACTERIA]
    result = np.column_stack(columns)
    if result.shape != (len(features), len(BACTERIA)) or not np.all(np.isfinite(result)):
        raise RuntimeError("MIC regressors returned invalid predictions")
    return result


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            torch.linspace(
                0,
                torch.log(torch.tensor(10000.0, device=times.device)),
                half,
                device=times.device,
            )
        )
        arguments = times[:, None] * frequencies[None, :]
        embedding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=-1)
        if embedding.shape[1] < self.dim:
            embedding = torch.cat(
                [embedding, torch.zeros((embedding.size(0), 1), device=times.device)], dim=-1
            )
        return self.mlp(embedding)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.ff(self.norm(values))


class LatentFlowNet(nn.Module):
    def __init__(self, dim: int, depth: int = 6, hidden: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(dim, hidden, dropout) for _ in range(depth)]
        )
        self.out = nn.Linear(dim, dim)

    def forward(self, values: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        hidden = values + self.time_emb(times).unsqueeze(1)
        for block in self.blocks:
            hidden = block(hidden)
        return self.out(hidden)


class AdapterHead(nn.Module):
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, values: torch.Tensor):
        hidden = self.proj(self.ln(values))
        return self.lm_head(hidden), hidden


def load_flow_generator(path: Path, device: torch.device):
    checkpoint = _torch_load(path, map_location=device)
    for key in ("dim", "flow", "adapter"):
        if key not in checkpoint:
            raise KeyError("Flow checkpoint is missing {!r}".format(key))
    dim = int(checkpoint["dim"])
    vocab_size = int(checkpoint.get("vocab_size", len(VOCAB)))
    flow = LatentFlowNet(dim=dim).to(device)
    adapter = AdapterHead(dim=dim, vocab_size=vocab_size).to(device)
    flow.load_state_dict(checkpoint["flow"], strict=True)
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    flow.eval()
    adapter.eval()
    return flow, adapter, dim, vocab_size


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned or "sequence"
