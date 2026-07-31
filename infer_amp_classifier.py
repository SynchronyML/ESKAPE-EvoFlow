#!/usr/bin/env python3
"""Predict AMP probabilities for peptide sequences."""

import argparse
from pathlib import Path

import pandas as pd

from evoflow_core import (
    ESMC_WEIGHTS_HELP,
    ESMCEncoder,
    load_amp_classifier,
    load_sequence_records,
    predict_amp,
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the ESKAPE-EvoFlow AMP classifier")
    parser.add_argument("--input", type=Path, help="CSV, TSV, FASTA or text input")
    parser.add_argument("--sequence", nargs="*", help="Peptide sequences supplied directly")
    parser.add_argument("--sequence-column", default="Sequence")
    parser.add_argument("--output", type=Path, default=Path("amp_predictions.csv"))
    parser.add_argument("--weight-dir", type=Path, default=Path("weight"))
    parser.add_argument(
        "--esmc-weights",
        type=Path,
        default=None,
        help=ESMC_WEIGHTS_HELP,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_sequence_records(args.input, args.sequence, args.sequence_column)
    identifiers, sequences = zip(*records)
    encoder = ESMCEncoder(args.esmc_weights, args.device)
    features = encoder.encode(sequences, batch_size=args.batch_size)
    device = resolve_device(args.device)
    model = load_amp_classifier(args.weight_dir / "amp_classifier.pt", device)
    probabilities = predict_amp(model, features, device)

    result = pd.DataFrame(
        {
            "ID": identifiers,
            "Sequence": sequences,
            "AMP_probability": probabilities,
            "AMP_prediction": (probabilities >= args.threshold).astype(int),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print("Saved {} predictions to {}".format(len(result), args.output))


if __name__ == "__main__":
    main()
