# Decoy File Generation for Data Flooding Against Ransomware

## Problem
Ransomware can ignore naive decoys when their statistical structure diverges from real user files.  
This project addresses that gap by generating **format‑aware, statistically plausible decoy files** that resemble real data distributions without copying content.  
The goal is to make data‑flooding defenses more effective by producing decoys that are harder to distinguish using standard statistical checks.

## Project Overview
It provides an end‑to‑end pipeline that **learns statistical models from real files**, **generates plausible decoys**, and **evaluates them quantitatively**. The output is a set of synthetic files and a set of metrics that demonstrate how closely they match real data by format.

## Detector
The **detector pipeline** provides an automated, repeatable way to **measure plausibility** and validate generated files against real datasets.  
At a high level it:
- Builds **statistical profiles** of real files using Byte Frequency Distribution (BFD) and N‑gram representations.
- Computes **format‑specific centroids** and evaluates distances between real and generated samples.
- Uses multiple **distance/divergence metrics** (entropy, JSD, TVD, cosine similarity, L1) to quantify similarity.
- Supports **train/test splitting** and **k‑fold evaluation** for consistent validation across formats.

In short, the detector is not a classifier that “flags ransomware.” It is an **evaluation framework** that scores how close synthetic files are to real ones, providing objective evidence of plausibility.

## Generator
The **generator** produces synthetic files that are statistically compatible with real data, without reusing original content.  
It operates by:
- Learning **Markov transition matrices** from real datasets per file format.
- Sampling those models to generate byte sequences with similar statistical structure.
- Supporting **format‑aware pipelines** (e.g., DOCX structure, PDF structure, JPEG markers) to preserve plausibility.

The result is a collection of decoys that better resemble real files than purely random output, making them more suitable for data‑flooding defenses.

## Key Results
- Generated files are **consistently closer to real files** than naive random decoys across multiple metrics (entropy, JSD, TVD, cosine similarity, L1).
- The detector pipeline provides **format‑specific evaluation** and can be run repeatedly to compare models, datasets, and parameter settings.
- The approach enables **objective validation** of plausibility instead of relying on visual inspection or ad‑hoc checks.

## Tech Stack / Methods
- **Language**: Python  
- **Models**: Markov transition models per format  
- **Statistics**: BFD and N‑gram representations  
- **Evaluation**: entropy, Jensen–Shannon divergence, total variation distance, cosine similarity, L1 distance

## Setup
From a fresh clone:

```bash
cd project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Install the optional Magika dependency only if you need the Magika analysis script:

```bash
python -m pip install -e ".[analysis]"
```

The Python packages live under `project/src`, and the project can be executed either as installed modules:

```bash
python -m detector.data.train_test_split --help
python -m generator.runtime.generator_finale --help
```

or as direct scripts from the repository root:

```bash
python3 project/src/detector/data/train_test_split.py --help
python3 project/src/generator/runtime/generator_finale.py --help
```

## Data And Models
Datasets, generated files, intermediate CSV/JSON files, and Markov matrices are intentionally not committed. Recreate this local layout before running the full pipelines:

```text
project/data/detector/datasets/
project/data/detector/derived/
project/data/detector/csv_utils/
project/data/generator/generated_files/
project/data/generator/matrices/
```

Expected detector dataset folders:

```text
project/data/detector/datasets/pdf data/PDF-total/
project/data/detector/datasets/pdf data/pdf_ranflood/
project/data/detector/datasets/txt data/TXT-total/
project/data/detector/datasets/txt data/txt_ranflood/
project/data/detector/datasets/jpg data/JPG-total/
project/data/detector/datasets/jpg data/jpg_ranflood/
project/data/detector/datasets/docx data/DOCX-total/
project/data/detector/datasets/docx data/docx_ranflood/
```

Generator runtime scripts require Markov matrices under `project/data/generator/matrices/`. Build them with the offline scripts before generating files.

Typical generator flow:

```bash
python -m generator.offline.build_txt_markov
python -m generator.offline.build_pdf_markov
python -m generator.offline.build_docx_markov
python -m generator.offline.build_jpg_markov
python -m generator.markov.build_alias_tables
python -m generator.markov.build_alias_jpg
python -m generator.runtime.generator_finale --num_files 1000
```

Typical detector flow:

```bash
python -m detector.data.kfold_split --out data/detector/derived/vari_json/json_split_dataset/kfold_split_3.json
python -m detector.scripts.generazione_soglie.generate_scores_kfold --out data/detector/csv_utils/csv_train_e_test_tutti_i_fold
python -m detector.scripts.generazione_soglie.optimize_thresholds_kfold --scores_dir data/detector/csv_utils/csv_train_e_test_tutti_i_fold --out_dir data/detector/csv_utils/soglie_ottimizzate_per_ogni_fold
```
