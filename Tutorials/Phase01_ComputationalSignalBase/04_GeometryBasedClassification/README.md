# 04 — Geometry-Based Classification of Bio-Images

## What you will learn

- The difference between **texture features** (how pixel intensities are distributed spatially) and **geometric / morphological features** (shape, area, perimeter, eccentricity)
- Computing **Gray-Level Co-occurrence Matrix (GLCM)** features: contrast, correlation, energy, homogeneity
- Extracting features with **Mahotas** and **scikit-image**
- Applying a classical classifier (e.g. SVM or k-NN) to the **HeLa cell** organelle dataset
- Understanding why these hand-crafted features were the state-of-the-art before deep learning — and where they still have advantages (small data, interpretability)

## Notebook

| File | Description |
|------|-------------|
| `geometry_based_classification.ipynb` | Full pipeline: dataset loading → feature extraction → classification → evaluation |

## Key concepts

**GLCM (Gray-Level Co-occurrence Matrix)** — encodes how often pairs of pixels with specific intensity values occur at a given spatial offset. Captures texture without any training.

**Morphological features** — area, perimeter, major/minor axis length, solidity, convex hull area. Derived from the binary mask of an object after segmentation.

**HeLa dataset** — fluorescence microscopy images of HeLa cells labelled with antibodies targeting specific organelles. A classic benchmark for subcellular localisation classification.

**Mahotas** — a fast C++ image-processing library with a clean Python API; particularly well-suited for bio-image feature extraction.

## Why this matters

Even in the age of deep learning, geometry-based features are used when:
- Only dozens or hundreds of labelled images are available
- A model must be interpretable for clinical or regulatory reasons
- Features are needed for downstream statistical analysis rather than end-to-end prediction

## Setup

```bash
pip install mahotas scikit-image scikit-learn numpy matplotlib
```

> **Troubleshooting:** if `from skimage.feature import graycomatrix` fails, upgrade scikit-image: `pip install --upgrade scikit-image`
