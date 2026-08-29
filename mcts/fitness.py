"""Frozen-oracle fitness types and the manuscript reward equation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


SPECIES = (
    "Enterococcus faecium",
    "Staphylococcus aureus",
    "Klebsiella pneumoniae",
    "Acinetobacter baumannii",
    "Pseudomonas aeruginosa",
    "Enterobacter cloacae",
)

MIC_COLUMNS = {
    "Enterococcus faecium": "log10_mic_E_faecium",
    "Staphylococcus aureus": "log10_mic_S_aureus",
    "Klebsiella pneumoniae": "log10_mic_K_pneumoniae",
    "Acinetobacter baumannii": "log10_mic_A_baumannii",
    "Pseudomonas aeruginosa": "log10_mic_P_aeruginosa",
    "Enterobacter cloacae": "log10_mic_E_cloacae",
}


def composite_fitness(
    amp_probability: Sequence[float] | np.ndarray,
    log10_mic: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the manuscript MIC activity component and composite reward.

    R(s) = 0.4 * pAMP(s) + 0.6 * mean_j(exp(-m_j(s))).
    The natural exponential is a monotonic activity surrogate, not 1/MIC.
    """

    amp = np.asarray(amp_probability, dtype=np.float64).reshape(-1)
    mic = np.asarray(log10_mic, dtype=np.float64)
    if mic.ndim == 1:
        mic = mic.reshape(1, -1)
    if mic.shape != (len(amp), len(SPECIES)):
        raise ValueError(
            "Expected MIC shape ({}, {}), got {}".format(
                len(amp), len(SPECIES), mic.shape
            )
        )
    if not np.all(np.isfinite(amp)) or not np.all(np.isfinite(mic)):
        raise ValueError("Fitness inputs must be finite")
    if np.any((amp < 0.0) | (amp > 1.0)):
        raise ValueError("AMP probabilities must lie in [0, 1]")
    mic_activity = np.exp(-mic).mean(axis=1)
    reward = 0.4 * amp + 0.6 * mic_activity
    if not np.all(np.isfinite(reward)):
        raise ValueError("Composite reward is not finite")
    return mic_activity, reward


@dataclass(frozen=True)
class Score:
    """Deterministic frozen-model score for one exact peptide sequence."""

    sequence: str
    amp_probability: float
    log10_mic: tuple[float, float, float, float, float, float]
    mic_activity_component: float
    composite_reward: float
    embedding: np.ndarray | None = None

    def as_row(self, include_sequence: bool = True) -> Dict[str, float | str]:
        row: Dict[str, float | str] = {}
        if include_sequence:
            row["sequence"] = self.sequence
        row["amp_probability"] = float(self.amp_probability)
        for species, value in zip(SPECIES, self.log10_mic):
            row[MIC_COLUMNS[species]] = float(value)
        row["mic_activity_component"] = float(self.mic_activity_component)
        row["composite_reward"] = float(self.composite_reward)
        return row

    @classmethod
    def from_values(
        cls,
        sequence: str,
        amp_probability: float,
        log10_mic: Iterable[float],
        embedding: np.ndarray | None = None,
    ) -> "Score":
        mic_tuple = tuple(float(value) for value in log10_mic)
        if len(mic_tuple) != len(SPECIES):
            raise ValueError("Exactly six MIC predictions are required")
        activity, reward = composite_fitness([amp_probability], [mic_tuple])
        return cls(
            sequence=sequence,
            amp_probability=float(amp_probability),
            log10_mic=mic_tuple,  # type: ignore[arg-type]
            mic_activity_component=float(activity[0]),
            composite_reward=float(reward[0]),
            embedding=None if embedding is None else np.asarray(embedding, dtype=np.float32),
        )
