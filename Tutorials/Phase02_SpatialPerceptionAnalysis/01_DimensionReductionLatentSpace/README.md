# 01 — Dimensionality Reduction & Latent Space

## What you will learn

- Why high-dimensional image data is hard to work with directly (the "curse of dimensionality")
- **Principal Component Analysis (PCA)**: what principal components are, how they are computed via SVD, and what "explaining variance" means
- Why **normalising** your data before PCA is essential
- Applying PCA for **image compression and reconstruction**: how many components do you need to retain visual quality?
- Exploring the latent space: what do individual principal components look like?
- Using **scikit-learn's PCA** implementation on a real retinal vessel dataset (DRIVE)

## Notebook

| File | Description |
|------|-------------|
| `dimension_reduction_reconstruction.ipynb` | PCA theory, sklearn implementation, compression, and visual reconstruction on retinal images |

## Key concepts

**Principal Component Analysis (PCA)** — finds orthogonal directions (principal components) in feature space that capture the most variance. Projecting data onto the top-*k* components gives the best *k*-dimensional linear approximation.

**Explained variance ratio** — the fraction of total data variance captured by each component. Plotting the cumulative explained variance vs. number of components tells you how many to keep.

**Latent space** — the lower-dimensional representation that results from encoding high-dimensional data. A good latent space clusters similar data points together.

**DRIVE dataset** — the Digital Retinal Images for Vessel Extraction benchmark. 40 retinal fundus images with manual vessel segmentation labels. Used here purely as a source of realistic biological images for PCA experiments.

## Practical context

Dimensionality reduction is used throughout cryo-EM and microscopy pipelines: 2-D class averaging reduces particle images to a compact embedding for clustering; UMAP and t-SNE (extensions of this idea) are used to visualise large particle or cell datasets.

## Setup

```bash
pip install scikit-learn numpy matplotlib pillow
```
