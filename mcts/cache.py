"""Persistent deterministic sequence-score cache backed by SQLite."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from .fitness import Score


class SQLiteScoreCache:
    """Cache embeddings and frozen-model scores without storing tree statistics."""

    def __init__(
        self,
        path: Path,
        metadata: Mapping[str, object],
        reuse: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists()
        if existed and not reuse:
            raise FileExistsError(
                "Score cache already exists: {}. Pass --reuse-score-cache to verify and reuse it."
                .format(self.path)
            )
        self.connection = sqlite3.connect(str(self.path), timeout=60.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scores (
                sequence TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                amp_probability REAL NOT NULL,
                mic_0 REAL NOT NULL,
                mic_1 REAL NOT NULL,
                mic_2 REAL NOT NULL,
                mic_3 REAL NOT NULL,
                mic_4 REAL NOT NULL,
                mic_5 REAL NOT NULL,
                mic_activity_component REAL NOT NULL,
                composite_reward REAL NOT NULL
            )
            """
        )
        canonical_metadata = {
            str(key): json.dumps(value, sort_keys=True, separators=(",", ":"))
            for key, value in metadata.items()
        }
        existing_metadata = dict(self.connection.execute("SELECT key, value FROM metadata"))
        if existing_metadata:
            if existing_metadata != canonical_metadata:
                raise RuntimeError(
                    "Score-cache metadata does not match the current models/configuration"
                )
        else:
            self.connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                sorted(canonical_metadata.items()),
            )
            self.connection.commit()
        self.memory: Dict[str, Score] = {}
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def get_many(self, sequences: Sequence[str]) -> Dict[str, Score]:
        unique = list(dict.fromkeys(sequences))
        found: Dict[str, Score] = {}
        missing_from_memory: list[str] = []
        for sequence in unique:
            if sequence in self.memory:
                found[sequence] = self.memory[sequence]
                self.hits += 1
            else:
                missing_from_memory.append(sequence)
        for start in range(0, len(missing_from_memory), 500):
            chunk = missing_from_memory[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            if not placeholders:
                continue
            rows = self.connection.execute(
                "SELECT sequence, embedding, embedding_dim, amp_probability, "
                "mic_0, mic_1, mic_2, mic_3, mic_4, mic_5, "
                "mic_activity_component, composite_reward FROM scores "
                "WHERE sequence IN ({})".format(placeholders),
                chunk,
            ).fetchall()
            for row in rows:
                embedding = np.frombuffer(row[1], dtype=np.float32, count=int(row[2])).copy()
                score = Score(
                    sequence=row[0],
                    embedding=embedding,
                    amp_probability=float(row[3]),
                    log10_mic=tuple(float(value) for value in row[4:10]),  # type: ignore[arg-type]
                    mic_activity_component=float(row[10]),
                    composite_reward=float(row[11]),
                )
                self.memory[score.sequence] = score
                found[score.sequence] = score
                self.hits += 1
        unresolved = len(unique) - len(found)
        self.misses += unresolved
        return found

    def put_many(self, scores: Iterable[Score]) -> None:
        rows = []
        for score in scores:
            if score.embedding is None:
                raise ValueError("Persistent cache requires the exact pooled embedding")
            embedding = np.asarray(score.embedding, dtype=np.float32).reshape(-1)
            rows.append(
                (
                    score.sequence,
                    sqlite3.Binary(embedding.tobytes(order="C")),
                    int(embedding.size),
                    float(score.amp_probability),
                    *[float(value) for value in score.log10_mic],
                    float(score.mic_activity_component),
                    float(score.composite_reward),
                )
            )
            self.memory[score.sequence] = score
        if rows:
            self.connection.executemany(
                "INSERT OR IGNORE INTO scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self.connection.commit()

    def close(self) -> None:
        self.connection.close()


class MemoryScoreCache:
    """In-memory cache used by unit tests and explicitly cache-free runs."""

    def __init__(self) -> None:
        self.memory: Dict[str, Score] = {}
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def get_many(self, sequences: Sequence[str]) -> Dict[str, Score]:
        found = {value: self.memory[value] for value in dict.fromkeys(sequences) if value in self.memory}
        self.hits += len(found)
        self.misses += len(set(sequences)) - len(found)
        return found

    def put_many(self, scores: Iterable[Score]) -> None:
        for score in scores:
            self.memory[score.sequence] = score

    def close(self) -> None:
        return None
