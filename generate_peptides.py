#!/usr/bin/env python3
"""Generate and rank peptides with the released rectified-flow checkpoint."""

import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from evoflow_core import (
    AA_ALPHABET,
    BACTERIA,
    CANONICAL_TOKEN_END,
    CANONICAL_TOKEN_START,
    load_amp_classifier,
    load_flow_generator,
    load_mic_models,
    predict_mics,
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ESKAPE-EvoFlow peptides")
    parser.add_argument("--weight-dir", type=Path, default=Path("weight"))
    parser.add_argument("--output", type=Path, default=Path("generated_peptides.csv"))
    parser.add_argument("--total-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--min-length", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def lhs_normal(batch_size: int, length: int, dim: int, temperature: float) -> torch.Tensor:
    coordinates = length * dim
    offsets = torch.rand(batch_size, coordinates)
    ranks = torch.argsort(torch.rand(batch_size, coordinates), dim=0).float()
    uniform = ((ranks + offsets) / batch_size).clamp(1e-6, 1.0 - 1e-6)
    normal = np.sqrt(2.0) * torch.erfinv(2.0 * uniform - 1.0)
    return normal.view(batch_size, length, dim) * temperature


def amp_gradient(values, classifier, scale):
    with torch.enable_grad():
        inputs = values.detach().requires_grad_(True)
        logits = classifier(inputs.mean(dim=1))
        energy = -F.logsigmoid(logits).mean() * scale
        gradient = torch.autograd.grad(energy, inputs)[0]
        norm = gradient.reshape(len(inputs), -1).norm(dim=1, keepdim=True).unsqueeze(-1)
        return gradient / (norm + 1e-8) * torch.clamp(norm, max=1.0)


@torch.no_grad()
def decode_canonical(adapter, values) -> List[str]:
    logits, _ = adapter(values)
    canonical = logits[:, :, CANONICAL_TOKEN_START:CANONICAL_TOKEN_END]
    indices = canonical.argmax(dim=-1).cpu().numpy()
    return ["".join(AA_ALPHABET[index] for index in row) for row in indices]


def main():
    args = parse_args()
    if args.min_length < 1 or args.max_length < args.min_length:
        raise ValueError("Invalid generation length range")
    if args.total_samples < 1 or args.batch_size < 1 or args.steps < 1:
        raise ValueError("Sample count, batch size and steps must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    flow, adapter, dim, _ = load_flow_generator(
        args.weight_dir / "flow_generator.pt", device
    )
    classifier = load_amp_classifier(args.weight_dir / "amp_classifier.pt", device)
    mic_models = load_mic_models(args.weight_dir)

    rows = []
    generated = 0
    while generated < args.total_samples:
        current_batch = min(args.batch_size, args.total_samples - generated)
        length = random.randint(args.min_length, args.max_length)
        values = lhs_normal(current_batch, length, dim, args.temperature).to(device)
        step_size = 1.0 / args.steps

        for step in range(args.steps):
            times = torch.full((current_batch,), step / args.steps, device=device)
            with torch.no_grad():
                velocity = flow(values, times)
            guidance = amp_gradient(values, classifier, args.guidance_scale)
            values = values + step_size * velocity - step_size * guidance

        sequences = decode_canonical(adapter, values)
        with torch.no_grad():
            pooled = values.mean(dim=1)
            amp_probabilities = torch.sigmoid(classifier(pooled)).cpu().numpy()
            features = pooled.float().cpu().numpy()
        mics = predict_mics(mic_models, features)
        scores = 10.0 * amp_probabilities - mics.sum(axis=1)

        for index, sequence in enumerate(sequences):
            row = {
                "Sequence": sequence,
                "Length": len(sequence),
                "Screening_score": float(scores[index]),
                "AMP_probability": float(amp_probabilities[index]),
            }
            for column, bacterium in enumerate(BACTERIA):
                row["{}_log10_MIC".format(bacterium)] = float(mics[index, column])
            rows.append(row)
        generated += current_batch

    result = pd.DataFrame(rows).sort_values("Screening_score", ascending=False)
    result = result.head(min(args.top_k, len(result))).reset_index(drop=True)
    result.insert(0, "Rank", np.arange(1, len(result) + 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print("Saved {} ranked peptides to {}".format(len(result), args.output))


if __name__ == "__main__":
    main()
