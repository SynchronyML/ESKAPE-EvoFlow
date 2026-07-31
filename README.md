# ESKAPE-EvoFlow

[English](README.md) | [简体中文](README.zh-CN.md)

ESKAPE-EvoFlow is an inference-only toolkit for antimicrobial peptide (AMP) classification, six-species MIC regression, rectified-flow peptide generation, and reward-guided peptide self-evolution. Training scripts, training datasets, and experimental result tables are not included.

> **Scope:** All outputs are computational predictions for sequence prioritization. They are not substitutes for experimental AMP or MIC measurements.

## 1. Configuration

### 1.1 Repository structure

```text
git-ESKAPE-EvoFlow/
├── evoflow_core.py             # Shared model definitions and I/O utilities
├── infer_amp_classifier.py     # AMP classifier inference
├── infer_mic_regressors.py     # Six-species MIC inference
├── generate_peptides.py        # Rectified-flow generation and ranking
├── self_evolution.py           # Local-mutation peptide self-evolution
├── requirements.txt
├── README.md
├── README.zh-CN.md
└── weight/                     # Local model files; ignored by Git
```

### 1.2 Runtime requirements

The public scripts were validated in the local `deepflavor` Mamba environment with the following configuration:

| Component | Validated version |
| --- | --- |
| Platform | WSL2, Ubuntu 24.04.2 LTS, x86-64 |
| Python | 3.13.3 |
| NumPy | 2.3.0 |
| pandas | 2.3.0 |
| SciPy | 1.16.2 |
| PyTorch | 2.7.1+cu126 |
| scikit-learn | 1.8.0 |
| joblib | 1.5.3 |
| EvolutionaryScale `esm` | 3.2.1.post1 |
| GPU used for validation | NVIDIA GeForce RTX 3090 Ti, 24 GB |
| NVIDIA driver / PyTorch CUDA build | 591.86 / CUDA 12.6 |

The project-relevant Python runtime stack is pinned in `requirements.txt`. The six MIC checkpoints carry the scikit-learn 1.8.0 serialization version, so that pin must be retained for reliable model loading. The scripts require the `ESMC` and `EsmSequenceTokenizer` interfaces provided by `esm==3.2.1.post1`.

CUDA is not required for parsing inputs or loading the lightweight predictors, but a CUDA-enabled PyTorch installation is strongly recommended for ESM C-600M encoding, rectified-flow generation, and self-evolution. CPU execution is supported by the CLI but was not used for the full validation run and will be substantially slower. The validated GPU had 24 GB of memory; this records the tested configuration rather than a formal minimum VRAM requirement.

### 1.3 Installation

Create a clean Mamba environment matching the validated Python version:

```bash
git clone <repository-url>
cd git-ESKAPE-EvoFlow

mamba create -n eskape-evoflow python=3.13.3 -y
mamba activate eskape-evoflow
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The scripts were also tested directly in the existing local environment:

```bash
mamba activate deepflavor
python --version
python -m pip install -r requirements.txt
```

For GPU execution, confirm that the installed PyTorch build can access CUDA before running ESM C inference:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### 1.4 Model weights

The scripts expect the following local layout:

```text
weight/
├── amp_classifier.pt
├── flow_generator.pt
├── MIC_Acinetobacter_baumannii.joblib
├── MIC_Enterobacter_cloacae.joblib
├── MIC_Enterococcus_faecium.joblib
├── MIC_Klebsiella_pneumoniae.joblib
├── MIC_Pseudomonas_aeruginosa.joblib
├── MIC_Staphylococcus_aureus.joblib
└── esmc-600m-2024-12/
    ├── config.json
    └── data/weights/esmc_600m_2024_12_v0.pth
```

Model binaries are intentionally excluded by `.gitignore` and must be obtained separately. `flow_generator.pt` contains the velocity network and the 26-token latent decoder. The ESM C checkpoint must be used under its original distribution terms. See [`weight/README.md`](weight/README.md) for details.

Verify the local model files against the included checksum manifest:

```bash
cd weight
sha256sum -c weights_manifest.sha256
```

Use `--weight-dir` to relocate the project predictors and `--esmc-weights` to provide either the ESM C model directory or its checkpoint file directly.

### 1.5 Input formats

Sequence-based commands accept:

- CSV or TSV containing a `Sequence` column;
- FASTA files (`.fa`, `.fasta`, or `.faa`);
- plain text with one sequence per line;
- one or more sequences supplied through `--sequence`.

Use `--sequence-column` to select a different table column. Sequences are uppercased and whitespace is removed. Any residue outside the 20 canonical amino acids raises an error; sequences are not silently modified.

Example CSV:

```csv
ID,Sequence
peptide_1,GRPPRHRIPPPRRVRVHPRF
peptide_2,KRWKFRQWWRMHWRRKCHKW
```

### 1.6 Representation and reproducibility

Classifier, MIC, and self-evolution inference use 1,152-dimensional ESM C-600M representations averaged over all non-padding tokens. Non-padding special tokens remain in the mean. No centering, feature scaling, or L2 normalization is applied before the released predictors.

- `--seed` controls Python, NumPy, and PyTorch sampling in generation and deterministic per-seed mutation streams in self-evolution.
- CUDA kernels and dependency versions may affect exact floating-point output.
- Checkpoints are loaded strictly; missing or structurally incompatible model states cause immediate failure.
- Run any command with `--help` for its complete parameter list.

## 2. Functions

### 2.1 AMP classification

Predict AMP probability and a thresholded AMP/non-AMP label for each input sequence:

```bash
python infer_amp_classifier.py \
  --input peptides.csv \
  --sequence-column Sequence \
  --output results/amp_predictions.csv \
  --device cuda
```

Direct sequence input is also supported:

```bash
python infer_amp_classifier.py \
  --sequence GRPPRHRIPPPRRVRVHPRF KRWKFRQWWRMHWRRKCHKW \
  --output results/amp_predictions.csv
```

Output columns are `ID`, `Sequence`, `AMP_probability`, and `AMP_prediction`. The default classification threshold is 0.5 and can be changed with `--threshold`.

### 2.2 Six-species MIC regression

Predict `log10(MIC)` and its raw-scale conversion for six ESKAPE species:

```bash
python infer_mic_regressors.py \
  --input peptides.fasta \
  --output results/mic_predictions.csv \
  --device cuda
```

The six regressors cover:

1. *Enterococcus faecium*
2. *Staphylococcus aureus*
3. *Klebsiella pneumoniae*
4. *Acinetobacter baumannii*
5. *Pseudomonas aeruginosa*
6. *Enterobacter cloacae*

For each species, the output contains `<species>_log10_MIC` and `<species>_MIC`, where the latter equals `10 ** predicted_log10_MIC`. It also contains `mean_log10_MIC`. The archived training tables identify the target as `log10(MIC)` but do not establish a physical unit; do not label the converted values as micromolar, mg/L, or another unit without independent metadata.

### 2.3 Rectified-flow peptide generation

Generate canonical peptide sequences with batch-local Latin-hypercube initialization, explicit Euler integration, and AMP-classifier latent guidance:

```bash
python generate_peptides.py \
  --total-samples 1000 \
  --batch-size 100 \
  --top-k 500 \
  --min-length 6 \
  --max-length 50 \
  --steps 50 \
  --temperature 1.1 \
  --guidance-scale 3.0 \
  --output results/generated_peptides.csv \
  --device cuda
```

Decoder argmax is restricted to the 20 canonical amino-acid tokens. Candidates are ranked by:

```text
screening_score = 10 * AMP_probability
                  - sum(six predicted log10(MIC) values)
```

The initial ranking uses the mean-pooled terminal flow latent. Decoded peptides are not re-encoded with ESM C for this ranking step.

### 2.4 Peptide self-evolution

Optimize each seed peptide in an independent UCB-style search tree:

```bash
python self_evolution.py \
  --input seed_peptides.csv \
  --output-dir results/self_evolution \
  --iterations 1000 \
  --patience 150 \
  --batch-size 32 \
  --seed 42 \
  --device cuda
```

For a parent sequence of length `L`, each mutation changes a uniformly sampled number of positions in:

```text
1, ..., max(1, floor(L / 5))
```

Positions are sampled without replacement, and each selected residue is replaced uniformly by one of the other 19 canonical amino acids. Length is preserved. The script performs no insertion, deletion, crossover, or fully random same-length sequence jump.

Each candidate is re-encoded by frozen ESM C and scored as:

```text
reward = 0.4 * AMP_probability
       + 0.6 * mean(exp(-predicted_log10_MIC_j), j=1..6)
```

The command writes one `<seed>_history.csv` per seed and a combined `self_evolution_summary.csv`.

## 3. Citation and contact

### 3.1 Citation

Citation information will be added when the associated study is published.

### 3.2 Contact

- For software usage, bug reports, and reproducibility questions, please open a GitHub Issue in this repository.
- Academic correspondence and the corresponding-author email will be added before public release.
