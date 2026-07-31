# ESKAPE-EvoFlow

[English](README.md) | [简体中文](README.zh-CN.md)

[![许可证](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

ESKAPE-EvoFlow 是用于抗菌肽（AMP）分类、六种 ESKAPE 细菌 MIC 回归、rectified-flow 肽序列生成以及奖励引导肽序列自进化的推理工具。本仓库不包含训练脚本、训练数据集和实验结果表。

> **适用范围：** 本仓库输出均为用于序列优先级排序的计算预测，不能替代 AMP 活性或 MIC 的实验测量。

## 1. 配置信息

### 1.1 仓库结构

```text
git-ESKAPE-EvoFlow/
├── evoflow_core.py             # 公共模型定义、权重加载和输入输出工具
├── infer_amp_classifier.py     # AMP 分类器推理
├── infer_mic_regressors.py     # 六菌 MIC 回归推理
├── generate_peptides.py        # Rectified-flow 生成与候选排序
├── self_evolution.py           # 基于局部突变的肽序列自进化
├── requirements.txt
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── README.md
├── README.zh-CN.md
├── weight/                     # 项目自有模型文件；默认被 Git 忽略
└── external_models/            # 可选的离线 ESM C 模型；默认被 Git 忽略
```

### 1.2 运行环境

公开脚本已在本地 `deepflavor` Mamba 环境中完成验证，具体配置如下：

| 组件 | 已验证版本 |
| --- | --- |
| 平台 | WSL2、Ubuntu 24.04.2 LTS、x86-64 |
| Python | 3.13.3 |
| NumPy | 2.3.0 |
| pandas | 2.3.0 |
| SciPy | 1.16.2 |
| PyTorch | 2.7.1+cu126 |
| scikit-learn | 1.8.0 |
| joblib | 1.5.3 |
| EvolutionaryScale `esm` | 3.2.1.post1 |
| huggingface-hub | 0.33.0 |
| 验证所用 GPU | NVIDIA GeForce RTX 3090 Ti，24 GB |
| NVIDIA 驱动 / PyTorch CUDA 构建 | 591.86 / CUDA 12.6 |

`requirements.txt` 固定了与本项目相关的 Python 运行时依赖。六个 MIC 权重均带有 scikit-learn 1.8.0 序列化版本标记，因此应保留该固定版本，以保证模型可靠加载。脚本依赖 `esm==3.2.1.post1` 提供的 `ESMC` 和 `EsmSequenceTokenizer` 接口。

解析输入和加载轻量级预测器并不强制要求 CUDA，但 ESM C-600M 编码、rectified-flow 生成及自进化强烈建议使用支持 CUDA 的 PyTorch。CLI 支持 CPU 设备，但完整验证未使用 CPU，且其运行速度会明显更慢。已验证显卡具有 24 GB 显存，这只是已测试配置，并非正式的最低显存要求。

### 1.3 安装

建议创建与已验证 Python 版本一致的独立 Mamba 环境：

```bash
git clone <repository-url>
cd git-ESKAPE-EvoFlow

mamba create -n eskape-evoflow python=3.13.3 -y
mamba activate eskape-evoflow
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

脚本也已在现有本地环境中直接测试：

```bash
mamba activate deepflavor
python --version
python -m pip install -r requirements.txt
```

使用 GPU 前，应确认当前 PyTorch 构建能够访问 CUDA：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### 1.4 模型权重

脚本默认使用以下本地目录结构：

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

这八个项目自有模型文件由 `.gitignore` 排除并单独分发。`flow_generator.pt` 同时包含速度网络和 26-token 潜变量解码器。更多信息参见 [`weight/README.md`](weight/README.md)。

使用仓库提供的校验清单检查本地模型文件：

```bash
cd weight
sha256sum -c weights_manifest.sha256
```

ESM C-600M 是外部依赖，不包含在本仓库、项目权重包、Zenodo 归档或校验清单中。用户需要自行从 [Hugging Face 官方仓库](https://huggingface.co/biohub/esmc-600m-2024-12)获取 `esmc-600m-2024-12` / `esmc_600m_2024_12_v0`。脚本默认调用 `ESMC.from_pretrained("esmc_600m")`，由 `esm==3.2.1.post1` 通过 Hugging Face 官方缓存自动下载或复用该模型。

官方模型卡目前同时标注 `MIT` 和 `other`，其中 `other` 指向 [`Biohub/esm` 第三方声明](https://github.com/Biohub/esm/blob/main/THIRD_PARTY_NOTICE.md)；官方 ESM 源码仓库按 [MIT 许可证](https://github.com/Biohub/esm/blob/main/LICENSE.md)发布。使用前应以官方模型卡和第三方声明的最新内容为准。

可通过 `--weight-dir` 指定八个项目预测器的目录。离线部署时，建议将官方 ESM C 仓库下载到固定位置 `external_models/esmc-600m-2024-12/`：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='biohub/esmc-600m-2024-12', local_dir='external_models/esmc-600m-2024-12')"
```

脚本预期的权重路径为 `external_models/esmc-600m-2024-12/data/weights/esmc_600m_2024_12_v0.pth`。运行任何依赖 ESM C 的命令时，通过 `--esmc-weights` 指定该模型目录：

```bash
python infer_amp_classifier.py \
  --input peptides.csv \
  --esmc-weights external_models/esmc-600m-2024-12 \
  --output results/amp_predictions.csv
```

### 1.5 输入格式

涉及序列输入的命令支持：

- 含 `Sequence` 列的 CSV 或 TSV；
- FASTA 文件（`.fa`、`.fasta` 或 `.faa`）；
- 每行一条序列的纯文本文件；
- 通过 `--sequence` 直接提供的一条或多条序列。

如果表格使用其他列名，可通过 `--sequence-column` 指定。脚本会将序列转换为大写并删除空白字符。若存在 20 种标准氨基酸以外的残基，脚本会直接报错，不会静默修改序列。

CSV 示例：

```csv
ID,Sequence
peptide_1,GRPPRHRIPPPRRVRVHPRF
peptide_2,KRWKFRQWWRMHWRRKCHKW
```

### 1.6 表征与可复现性

分类、MIC 回归和自进化推理使用官方 ESM C-600M tokenizer 及模型配置：36 个 Transformer 层、1,152 维隐藏表示和 18 个 attention heads。表征对全部 non-padding token 求平均，并保留 non-padding 的特殊 token。输入发布的预测器前不进行额外中心化、特征缩放或 L2 归一化。

- `--seed` 控制生成流程中的 Python、NumPy 和 PyTorch 随机数，并为每条初始肽分配确定性的突变随机流。
- CUDA 内核及依赖版本可能导致浮点结果存在细微差异。
- 权重采用严格加载；文件缺失或模型结构不兼容时立即失败。
- 所有命令均可添加 `--help` 查看完整参数列表。

### 1.7 许可证

ESKAPE-EvoFlow 源代码及八个项目自有模型权重均按 [Apache License 2.0](LICENSE) 发布。将项目自有权重上传至 Zenodo 时，应在记录元数据中选择 `Apache-2.0`，并在可下载的权重归档中包含一份 `LICENSE`。

本项目许可证不覆盖 ESM C 或其他第三方依赖。ESM C 由用户从 Biohub 单独获取，并继续受其上游模型卡、MIT 许可证和第三方声明约束。依赖边界和官方来源链接参见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 2. 功能说明

### 2.1 AMP 分类

预测每条输入序列的 AMP 概率以及经过阈值判定的 AMP/non-AMP 标签：

```bash
python infer_amp_classifier.py \
  --input peptides.csv \
  --sequence-column Sequence \
  --output results/amp_predictions.csv \
  --device cuda
```

也可直接输入序列：

```bash
python infer_amp_classifier.py \
  --sequence GRPPRHRIPPPRRVRVHPRF KRWKFRQWWRMHWRRKCHKW \
  --output results/amp_predictions.csv
```

输出列包括 `ID`、`Sequence`、`AMP_probability` 和 `AMP_prediction`。默认分类阈值为 0.5，可通过 `--threshold` 修改。

### 2.2 六菌 MIC 回归

预测六种 ESKAPE 细菌的 `log10(MIC)` 及其还原到原始尺度后的数值：

```bash
python infer_mic_regressors.py \
  --input peptides.fasta \
  --output results/mic_predictions.csv \
  --device cuda
```

六个回归器分别对应：

1. *Enterococcus faecium*
2. *Staphylococcus aureus*
3. *Klebsiella pneumoniae*
4. *Acinetobacter baumannii*
5. *Pseudomonas aeruginosa*
6. *Enterobacter cloacae*

每种菌对应输出 `<species>_log10_MIC` 和 `<species>_MIC` 两列，后者等于 `10 ** predicted_log10_MIC`；另输出 `mean_log10_MIC`。现有训练表只能确认目标值为 `log10(MIC)`，无法确认物理单位。因此，在没有独立元数据支持时，不能将还原后的数值标注为 μM、mg/L 或其他单位。

### 2.3 Rectified-flow 肽序列生成

使用 batch 内独立的拉丁超立方初始化、显式 Euler 积分和 AMP 分类器潜变量引导生成仅含标准氨基酸的肽序列：

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

解码器的 argmax 被限制在 20 种标准氨基酸 token 中。候选序列按以下公式排序：

```text
screening_score = 10 * AMP_probability
                  - sum(六个预测 log10(MIC))
```

初筛使用终点 flow 潜变量的平均池化表征；此排序步骤不会使用 ESM C 重新编码已经解码的肽序列。

### 2.4 肽序列自进化

在独立的 UCB 风格搜索树中优化每条初始肽：

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

对于长度为 `L` 的父序列，每次突变从以下整数范围均匀抽取突变位点数：

```text
1, ..., max(1, floor(L / 5))
```

突变位置无放回抽样，每个位点均匀替换为其余 19 种标准氨基酸之一。序列长度保持不变。脚本不执行插入、缺失、交叉，也不生成完全随机的同长度序列。

每条候选序列都会由冻结的 ESM C 重新编码，并按以下公式评分：

```text
reward = 0.4 * AMP_probability
       + 0.6 * mean(exp(-predicted_log10_MIC_j), j=1..6)
```

命令为每条初始肽输出一个 `<seed>_history.csv`，并生成汇总文件 `self_evolution_summary.csv`。

## 3. 引用和联系方式

### 3.1 引用

相关研究发表后将在此补充引用信息。

### 3.2 联系方式

- 软件使用、错误报告和可复现性问题请通过本仓库的 GitHub Issues 提交。
- 学术通讯及通讯作者邮箱将在仓库公开发布前补充。
