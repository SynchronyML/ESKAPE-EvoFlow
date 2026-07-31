#!/usr/bin/env python3
"""Predict six ESKAPE log10(MIC) values for peptide sequences."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evoflow_core import (
    BACTERIA,
    ESMCEncoder,
    load_mic_models,
    load_sequence_records,
    predict_mics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run six ESKAPE MIC regressors")
    parser.add_argument("--input", type=Path, help="CSV, TSV, FASTA or text input")
    parser.add_argument("--sequence", nargs="*", help="Peptide sequences supplied directly")
    parser.add_argument("--sequence-column", default="Sequence")
    parser.add_argument("--output", type=Path, default=Path("mic_predictions.csv"))
    parser.add_argument("--weight-dir", type=Path, default=Path("weight"))
    parser.add_argument(
        "--esmc-weights",
        type=Path,
        default=Path("weight/esmc-600m-2024-12"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_sequence_records(args.input, args.sequence, args.sequence_column)
    identifiers, sequences = zip(*records)
    encoder = ESMCEncoder(args.esmc_weights, args.device)
    features = encoder.encode(sequences, batch_size=args.batch_size)
    models = load_mic_models(args.weight_dir)
    predicted_log10 = predict_mics(models, features)

    result = pd.DataFrame({"ID": identifiers, "Sequence": sequences})
    for index, bacterium in enumerate(BACTERIA):
        result["{}_log10_MIC".format(bacterium)] = predicted_log10[:, index]
        result["{}_MIC".format(bacterium)] = np.power(10.0, predicted_log10[:, index])
    result["mean_log10_MIC"] = predicted_log10.mean(axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print("Saved {} predictions to {}".format(len(result), args.output))


if __name__ == "__main__":
    main()
