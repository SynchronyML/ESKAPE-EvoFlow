# ESKAPE-EvoFlow

[English](README.md) | [简体中文](README.zh-CN.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21723051.svg)](https://doi.org/10.5281/zenodo.21723051)

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
├── self_evolution.py           # Legacy lightweight local-mutation search
├── mcts/                       # Auditable UCT policy, scoring, cache, and tree state
├── scripts/
│   ├── run_uct_mcts.py         # Formal manuscript-exact PureMCTS/MixedMCTS entrypoint
│   └── audit_manuscript_mcts.py # Regenerate/check the consistency audit
├── configs/uct_mcts_manuscript.yaml
├── tests/                      # MCTS formula, constraint, resume, and determinism tests
├── MCTS_IMPLEMENTATION_AUDIT.md
├── MANUSCRIPT_MCTS_CONSISTENCY_AUDIT.md
├── requirements.txt
├── LICENSE
├── README.md
├── README.zh-CN.md
├── weight/                     # Project-owned model files; ignored by Git
└── external_models/            # Optional offline ESM C model; ignored by Git
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
| huggingface-hub | 0.33.0 |
| RDKit | 2025.03.2 |
| PyYAML | 6.0.2 |
| tqdm | 4.67.1 |
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
└── MIC_Staphylococcus_aureus.joblib
```

These eight project-owned model files are intentionally excluded by `.gitignore` and distributed separately through [Zenodo (DOI: 10.5281/zenodo.21723051)](https://doi.org/10.5281/zenodo.21723051). `flow_generator.pt` contains the velocity network and the 26-token latent decoder. See [`weight/README.md`](weight/README.md) for details.

Verify the local model files against the included checksum manifest:

```bash
cd weight
sha256sum -c weights_manifest.sha256
```

ESM C-600M is an external dependency and is not included in this repository, the project weight package, Zenodo archive, or checksum manifest. Users must obtain `esmc-600m-2024-12` / `esmc_600m_2024_12_v0` from the [official Hugging Face repository](https://huggingface.co/biohub/esmc-600m-2024-12). By default, the scripts call `ESMC.from_pretrained("esmc_600m")`, allowing `esm==3.2.1.post1` to download or reuse the model through the official Hugging Face cache.

The official model card currently declares `MIT` and `other`, with `other` referring to the [`Biohub/esm` third-party notice](https://github.com/Biohub/esm/blob/main/THIRD_PARTY_NOTICE.md); the official ESM source repository provides its code under the [MIT license](https://github.com/Biohub/esm/blob/main/LICENSE.md). Users must review the current upstream model card and notices before use.

Use `--weight-dir` to relocate the eight project predictors. For a reproducible offline layout, download the official ESM C repository into `external_models/esmc-600m-2024-12/`:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='biohub/esmc-600m-2024-12', local_dir='external_models/esmc-600m-2024-12')"
```

The expected checkpoint path is `external_models/esmc-600m-2024-12/data/weights/esmc_600m_2024_12_v0.pth`. Pass that model directory to any ESM C-dependent command with `--esmc-weights`:

```bash
python infer_amp_classifier.py \
  --input peptides.csv \
  --esmc-weights external_models/esmc-600m-2024-12 \
  --output results/amp_predictions.csv
```

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

Classifier, MIC, and self-evolution inference use the official ESM C-600M tokenizer and model configuration: 36 transformer layers, hidden size 1,152, and 18 attention heads. Representations are averaged over all non-padding tokens; non-padding special tokens remain in the mean. No centering, feature scaling, or L2 normalization is applied before the released predictors.

- `--seed` controls Python, NumPy, and PyTorch sampling. UCT-MCTS derives an independent stable seed from SHA256(base seed, parent ID, strategy) for every tree.
- CUDA kernels and dependency versions may affect exact floating-point output.
- Checkpoints are loaded strictly; missing or structurally incompatible model states cause immediate failure.
- Run any command with `--help` for its complete parameter list.

### 1.7 License

The ESKAPE-EvoFlow source code and the eight project-owned model weights are released under the [Apache License 2.0](LICENSE). When depositing the project-owned weights on Zenodo, select `Apache-2.0` in the record metadata and include a copy of `LICENSE` in the downloadable archive.

This project license does not cover ESM C or other third-party dependencies. ESM C is not redistributed by this project: users obtain it separately from the [official Biohub model repository](https://huggingface.co/biohub/esmc-600m-2024-12), and its software, model, and dependencies remain subject to the upstream [MIT license](https://github.com/Biohub/esm/blob/main/LICENSE.md), model card, and [third-party notices](https://github.com/Biohub/esm/blob/main/THIRD_PARTY_NOTICE.md). Runtime packages listed in `requirements.txt` retain their respective upstream licenses.

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

The manuscript-exact implementation builds one independent UCT tree for each flow-generated parent and supports `PureMCTS` and `MixedMCTS`:

**Formal manuscript reproduction entrypoint:** `scripts/run_uct_mcts.py`. Run it from the repository root so the frozen config, `mcts` package and shared `evoflow_core.py` are resolved from this checkout. The older root-level `self_evolution.py` is not the manuscript reproduction entrypoint.

```bash
python scripts/run_uct_mcts.py \
  --parents flow_generated_parents.csv \
  --strategies PureMCTS MixedMCTS \
  --devices cuda:0 \
  --seed 42 \
  --output-dir outputs/uct_mcts \
  --cache-dir outputs/uct_mcts/cache
```

For multiple GPUs, list each device once; one persistent worker and one resident model set are created per device:

```bash
python scripts/run_uct_mcts.py \
  --parents flow_generated_parents.csv \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
  --seed 42
```

The frozen defaults are in `configs/uct_mcts_manuscript.yaml`: branching factor 8, exploration coefficient 25, epsilon `1e-6`, exactly `min(150, 5L)` ordered proposal attempts per expansion, 1,000 expansions, and patience 150. Duplicates do not trigger refill attempts. PureMCTS uses 100% local proposals with one/two substitutions at probabilities 0.8/0.2. MixedMCTS dispatches each proposal event 50:50 between that local kernel and an independently sampled, canonical, same-length global peptide. Pure and Mixed otherwise use identical selection, scoring, insertion, backpropagation and stopping rules.

When a node already has eight children, traversal uses:

```text
UCT(a|s) = Q(a)/(N(a)+1e-6)
           + 25 * U(a; T excluding a)
             * sqrt(log(N(s)+1)/(N(a)+1e-6))
```

`U` is one minus the maximum Tanimoto similarity between the child's non-chiral binary radius-3, 1,024-bit Morgan fingerprint and all other nodes in the same tree. Excluding the child itself prevents the exploration term from collapsing to zero. See `MCTS_IMPLEMENTATION_AUDIT.md` for the source comparison and `MANUSCRIPT_MCTS_CONSISTENCY_AUDIT.md` for the automatically verified final checklist.

Every eligible candidate is re-encoded by frozen ESM C and scored in a batch with the SiLU AMP inference implementation and six independent MIC regressors:

```text
reward = 0.4 * AMP_probability
       + 0.6 * mean(exp(-predicted_log10_MIC_j), j=1..6)
```

Only one highest-immediate-reward candidate is inserted per expansion; exact ties select the first maximum in proposal-generation order. No independent stochastic rollout is performed. The new reward is back-propagated from child through root. Early stopping uses the literal comparison `R_new > R_best` with no tolerance. Sequence scores are cached independently from tree statistics. Each parent/strategy directory contains `nodes.csv`, `edges.csv`, `candidate_evaluations.csv`, `selection_log.csv`, `trajectory.csv`, `lineage.csv`, `final_result.csv`, `summary.json`, and an atomic resume checkpoint. Use `--resume --reuse-score-cache` after interruption.

Run the model-free contract tests with:

```bash
python -m unittest discover -s tests -v
```

Regenerate and verify the formal consistency checklist with:

```bash
python scripts/audit_manuscript_mcts.py
python scripts/audit_manuscript_mcts.py --check
```

Run the two-parent, 20-expansion-per-tree model smoke test before a full search:

```bash
python scripts/run_uct_mcts.py \
  --sequence KSYKFECRWRFHLTTNCIKT GRPPRHRIPPPRRVRVHPRF \
  --devices cuda:0 \
  --output-dir results/runtime_smoke_test/uct_mcts \
  --cache-dir results/runtime_smoke_test/uct_mcts/cache \
  --smoke-test
```

The implementation status recorded by the automated audit is `MANUSCRIPT_EXACT_UCT_MCTS_READY`.

## 3. Citation and contact

### 3.1 Citation

Citation information will be added when the associated study is published.

Project-owned model weights: Cui, Zhiyong. (2026). *ESKAPE-EvoFlow*. Zenodo. [https://doi.org/10.5281/zenodo.21723051](https://doi.org/10.5281/zenodo.21723051).

### 3.2 Contact

- For software usage, bug reports, and reproducibility questions, please open a GitHub Issue in this repository.
- Academic correspondence and the corresponding-author email will be added before public release.
