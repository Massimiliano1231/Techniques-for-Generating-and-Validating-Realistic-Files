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
