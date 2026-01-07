# FKAN-a: Fourier-enhanced Kolmogorov-Arnold Network for Drug-Target Interaction Prediction

FKAN-a is a deep learning framework for predicting drug-target interactions (DTI) by integrating Fourier spectral features and graph representations. This implementation supports multi-scale spectral fusion and handles both protein sequence embeddings and molecular graph features

## Features

Multi-scale Fourier spectral fusion for protein representation.

Kolmogorov-Arnold Network for drug-target interaction modeling.

Supports graph-based drug representation (e.g., molecular graphs) and protein embeddings (e.g., ESM or other pretrained embeddings).

Flexible training, evaluation, and inference scripts.

Implements standard DTI evaluation metrics: AUROC, AUPRC, Accuracy, Sensitivity, Specificity.

## Requirements

FKAN-a supports environment setup via Conda using the provided environment.yml.

To create the environment: 
```bash
conda env create -f environment.yml
conda activate fkan-a
```

## Training

To train FKAN-a using a config file:

```bash
python -m __main__ train \
    --run-id TestRun \
    --config config/default_config.yam
```
