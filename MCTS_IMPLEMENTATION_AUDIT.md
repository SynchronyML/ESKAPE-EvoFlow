# ESKAPE-EvoFlow UCT-MCTS implementation audit

## Scope and source priority

This audit records the correction from the first public implementation draft to the final frozen manuscript contract supplied on 2026-08-29. The latest author-supplied contract has highest priority. Historical evidence was checked in `../4_2_mcts_optimization.py`, `../outputs/manuscript/Methods_bilingual_CN_EN.md`, the revised manuscript document, and archived MCTS result directories. No historical result is treated as proof that the corrected implementation was used; formal results must be rerun.

## Exact final UCT equation

For a parent `s` with exactly eight children and child `a`:

```text
UCT(a | s) = Q(a) / [N(a) + 10^-6]
             + 25 × U(a; T \ {a})
               × sqrt(log[N(s) + 1] / [N(a) + 10^-6])
```

```text
U(a; T \ {a}) = 1 - max(q in T, q != a)
                     Tanimoto(Morgan_r=3,nBits=1024,includeChirality=False(a),
                               Morgan_r=3,nBits=1024,includeChirality=False(q)).
```

`T` contains only nodes already inserted into this one tree. The current child is excluded by node identity. If no other reference exists, uniqueness is 1.0.

## Manuscript-contract correction

| Item | Previous draft | Final implementation | Status |
| --- | --- | --- | --- |
| UCT exploitation | `Q/N` via `mean_reward` | Literal `Q/(N+1e-6)` | PASS after correction |
| Candidate generation | Could retry up to five batches after filtering | Exactly `min(150,5L)` ordered attempts, no refill | PASS after correction |
| MixedMCTS | Two labels dispatching to the same local kernel | 50% local plus 50% independently sampled same-length global peptide | PASS after correction |
| Reward ties | Reward then lexicographic sequence | First maximum in proposal-generation order | PASS after correction |
| Early stopping | `R_new > R_best + 1e-12` | Literal `R_new > R_best` | PASS after correction |

The former local-only MixedMCTS interpretation is superseded. The corrected Mixed kernel now matches the manuscript description that it expands exploration beyond the immediate mutational neighbourhood. Earlier smoke-test trajectories and any historical MixedMCTS results must not be combined with corrected results.

## Full contract comparison

| Item | Frozen contract | Final implementation | Status |
| --- | --- | --- | --- |
| Root | Flow-generated parent; `N=1`, `Q=R` | Exact | PASS |
| Selection trigger | Compute UCT only at exactly 8 children | Exact; more than 8 raises | PASS |
| Exploitation | `Q/(N+1e-6)` | Exact | PASS |
| Exploration | `25*U*sqrt(log(Nparent+1)/(Nchild+1e-6))` | Exact | PASS |
| Fingerprint | `MolFromSequence`; binary Morgan radius 3, 1024 bits, no chirality | Exact | PASS |
| Uniqueness | Other inserted nodes in this tree; exclude self | Exact | PASS |
| Attempts | Exactly `min(150,5L)` events | Exact; duplicates do not refill | PASS |
| PureMCTS | 100% local, 0.8 one-site/0.2 two-site | Exact | PASS |
| MixedMCTS | 50% local/50% independent same-length global | Exact | PASS |
| Alphabet and length | Canonical, fixed length | Exact | PASS |
| Candidate scoring | Fresh frozen ESM C encoding, SiLU AMP in eval mode, six MIC models | Exact | PASS |
| Fitness | `0.4*pAMP + 0.6*mean(exp(-log10MIC))` | Exact | PASS |
| Cache | May deduplicate inference only | Ordered proposal multiset retained; cache separate from tree state | PASS |
| Tree duplicate | Only already-inserted sequences prohibited | Exact | PASS |
| Insertion | One first-maximum candidate | Exact | PASS |
| Backpropagation | Add child immediate reward through root | Exact | PASS |
| Rollout | None | None | PASS |
| Stopping | 1000 expansions or 150 strict non-improvements | Exact, no tolerance | PASS |
| Tree independence | One tree per parent and strategy | Separate statistics, fingerprints, seen set, best and patience | PASS |
| Reproducibility | Deterministic seed, checkpoint/resume and audit outputs | Preserved | PASS |

## Checkpoint identities

| Model | Repository path | SHA256 |
| --- | --- | --- |
| ESM C-600M | `external_models/esmc-600m-2024-12/data/weights/esmc_600m_2024_12_v0.pth` | `8ef856e1a237ee3f995442df997a962e70057faadecf38fc0c8561bd3c2f4324` |
| AMP predictor | `weight/amp_classifier.pt` | `654cb4a3d8a958652098cc1c0abf52b9bac2536f6d5ee773564ecc9ae7487124` |
| *E. faecium* MIC | `weight/MIC_Enterococcus_faecium.joblib` | `566cac193415a4e2d204b75b62ce986f0a2af923e474810edf5da7f37907c836` |
| *S. aureus* MIC | `weight/MIC_Staphylococcus_aureus.joblib` | `2765fcc5dbafc2479576fec60156f8985578b6ed8a481b72cf2d267b2652c3b7` |
| *K. pneumoniae* MIC | `weight/MIC_Klebsiella_pneumoniae.joblib` | `16dfe52236880629a5ed912ab4dbe77d47a90e5e45118a32a83fce7ed08f1861` |
| *A. baumannii* MIC | `weight/MIC_Acinetobacter_baumannii.joblib` | `d34d750d2d9f9f6a2e7920346d0f4ef32ff139bdba4802a8da3216f08978ba02` |
| *P. aeruginosa* MIC | `weight/MIC_Pseudomonas_aeruginosa.joblib` | `17ace75e5a0f951b178dfb919445d2b40fea020900957abc0aaefa929470fde5` |
| *E. cloacae* MIC | `weight/MIC_Enterobacter_cloacae.joblib` | `1c129ab80911ed2ef6515d6685f0c99c80103d16ace941630966f912da9a0929` |

Project checkpoint binaries, the external ESM C model, caches and run outputs remain excluded from Git. The executable checklist is generated by `scripts/audit_manuscript_mcts.py` in `MANUSCRIPT_MCTS_CONSISTENCY_AUDIT.md`.
