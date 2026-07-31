# Project model weights

Only the eight project-owned inference files belong in this directory and in the corresponding Zenodo record:

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

These eight project-owned files are released under the repository [Apache License 2.0](../LICENSE). A standalone Zenodo weight archive must include a copy of `LICENSE`, and the Zenodo record license should be set to `Apache-2.0`.

The external ESM C-600M checkpoint is not distributed with this project or included in this checksum manifest. Users must obtain `esmc-600m-2024-12` / `esmc_600m_2024_12_v0` from the [official Hugging Face repository](https://huggingface.co/biohub/esmc-600m-2024-12). The scripts load the official model by default through `ESMC.from_pretrained("esmc_600m")` and its Hugging Face cache.

For offline use, place the downloaded official repository at `external_models/esmc-600m-2024-12/`, so the checkpoint is available at `external_models/esmc-600m-2024-12/data/weights/esmc_600m_2024_12_v0.pth`, and pass `--esmc-weights external_models/esmc-600m-2024-12`.

Verify the eight project files from this directory with:

```bash
sha256sum -c weights_manifest.sha256
```
