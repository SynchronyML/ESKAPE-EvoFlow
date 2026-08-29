from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from mcts.cache import MemoryScoreCache
from mcts.fitness import Score


class DeterministicMockScorer:
    def __init__(self):
        self.cache = MemoryScoreCache()
        self.metadata = {"backend": "deterministic-test-double"}
        self.new_score_count = 0

    def get_scores(self, sequences):
        unique = list(dict.fromkeys(sequences))
        found = self.cache.get_many(unique)
        generated = []
        for sequence in unique:
            if sequence in found:
                continue
            digest = hashlib.sha256(sequence.encode("ascii")).digest()
            amp = int.from_bytes(digest[:4], "big") / (2**32 - 1)
            mic = [2.0 * digest[index] / 255.0 - 1.0 for index in range(4, 10)]
            score = Score.from_values(
                sequence,
                amp,
                mic,
                embedding=np.frombuffer(digest, dtype=np.uint8).astype(np.float32),
            )
            generated.append(score)
            found[sequence] = score
            self.new_score_count += 1
        self.cache.put_many(generated)
        return found


def small_config(max_expansions=12):
    from mcts.search import SearchConfig

    return SearchConfig(
        max_expansions=max_expansions,
        patience=max_expansions,
        max_candidates=30,
        candidates_per_residue=3,
        checkpoint_interval=2,
    )
