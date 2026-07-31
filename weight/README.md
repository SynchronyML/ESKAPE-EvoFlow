# Model weights

Model binaries are local runtime dependencies and are excluded from Git by the repository `.gitignore`. Obtain them separately and preserve this layout:

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

The ESM C-600M checkpoint is an external pretrained model and must be obtained and used under its original distribution terms. If a `weights_manifest.sha256` file accompanies the repository, verify the local files from this directory with:

```bash
sha256sum -c weights_manifest.sha256
```
