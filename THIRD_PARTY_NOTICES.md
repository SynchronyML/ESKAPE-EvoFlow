# Third-party notices

ESKAPE-EvoFlow source code and the eight project-owned model weights are licensed under the Apache License 2.0. That license does not replace or modify the licenses of third-party software or models used by the project.

## ESM C-600M

ESKAPE-EvoFlow uses ESM C-600M as an external, frozen sequence encoder. The ESM C software and model checkpoint are not redistributed in this repository or in the project-owned weight package.

- Software: [`Biohub/esm`](https://github.com/Biohub/esm), provided under the [MIT License](https://github.com/Biohub/esm/blob/main/LICENSE.md).
- Model: [`biohub/esmc-600m-2024-12`](https://huggingface.co/biohub/esmc-600m-2024-12), whose model card identifies `MIT` and `other`; the linked `other` notice is the upstream [`THIRD_PARTY_NOTICE.md`](https://github.com/Biohub/esm/blob/main/THIRD_PARTY_NOTICE.md).

Users obtain ESM C directly from the official source and are responsible for complying with the current upstream model card, license, and third-party notices.

## Runtime dependencies

Packages listed in `requirements.txt` are installed separately and retain their respective upstream licenses. Their inclusion as dependencies does not place those third-party packages under the ESKAPE-EvoFlow Apache-2.0 license.
