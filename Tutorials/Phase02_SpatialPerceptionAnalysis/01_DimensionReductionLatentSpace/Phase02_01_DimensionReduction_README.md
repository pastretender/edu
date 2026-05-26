# 01 — Dimensionality Reduction & Latent Space

## Overview

A 512×512 fluorescence microscopy image contains 262,144 pixel values. Feeding that raw
vector directly into a classifier or clustering algorithm is almost always a bad idea: most
of those dimensions are correlated, redundant, or dominated by noise — and many algorithms
degrade sharply as dimensionality grows (the "curse of dimensionality"). The solution is to
find a compact, meaningful representation that preserves the structure you care about while
discarding noise.

This module teaches you to do that using **Principal Component Analysis (PCA)**, the most
widely used and mathematically transparent dimensionality-reduction method. By the end you
will understand PCA deeply enough to interpret what each component represents, diagnose
when it fails, and apply it correctly to biological image datasets.

---

## What you will learn

### Why high dimensions are a problem

As the number of dimensions $d$ grows, several things go wrong simultaneously:

- **Data sparsity:** the volume of a unit hypercube grows as $2^d$. With a fixed number of
  samples, data points become increasingly isolated — nearest neighbours are almost as far
  away as randomly chosen points.
- **Distance concentration:** in high dimensions, the ratio of the maximum to minimum
  pairwise distance tends to 1, making distance-based algorithms unreliable.
- **Visualisation:** you cannot plot more than 3 dimensions without a projection. A 2-D or
  3-D embedding is essential for any exploratory analysis.
- **Overfitting:** classifiers fitted in a high-dimensional space require exponentially
  more training samples to generalise — the regime where geometry-based features (Phase 01
  Module 04) or PCA are preferable to raw pixels.

### Principal Component Analysis

PCA finds a set of orthogonal directions (the **principal components**) in the original
feature space that capture the maximum variance. The data is then projected onto the top
$k$ components, producing a $k$-dimensional representation where $k \ll d$.

**The SVD connection:** PCA is computed by the Singular Value Decomposition of the
mean-centred data matrix $X$:

$$X = U \Sigma V^T$$

The principal components are the columns of $V$ (the right singular vectors). Projecting
onto the top $k$ components gives the best rank-$k$ linear approximation of $X$ in the
Frobenius norm — this is the Eckart-Young theorem.

**Explained variance:** the singular values $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_d$
measure how much variance each component captures. The fraction explained by component $i$ is:

$$r_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2}$$

Plotting the **cumulative explained variance** vs. number of components tells you how many
components are needed to retain 90%, 95%, or 99% of the total variance — the key decision
in dimensionality reduction.

### Why normalisation before PCA is mandatory

PCA finds directions of maximum variance. If one feature has values in the range [0,
10000] and another in [0, 1], the first feature will dominate the first principal component
regardless of its biological meaning. Normalising features to zero mean and unit variance
(`StandardScaler`) gives each feature an equal opportunity to contribute.

For images of the same type (same channel, same acquisition protocol), per-pixel
mean-subtraction (subtracting the dataset mean image) is the appropriate normalisation —
this removes the "average face" (or average cell) before looking for variation.

### Interpreting principal components visually

For image data, each principal component is a vector of the same dimensionality as the
original images — so it can be reshaped and displayed as an image. This is how you
understand *what* each component captures:

- **PC1** typically captures the largest source of variation in the dataset: global
  brightness, the presence or absence of a dominant structure, or coarse morphological
  shape
- **PC2** captures the second-largest source of variation orthogonal to PC1
- **Later PCs** capture progressively finer variation, eventually becoming dominated by
  noise

In the retinal vessel dataset used in this module, early PCs correspond to vessel density
gradients across the field of view, while later PCs capture fine branching structure.

### Image compression and reconstruction

PCA naturally provides a lossy compression scheme. Given the top $k$ components:

$$\hat{x} = \bar{x} + \sum_{i=1}^{k} (x - \bar{x})^T v_i \; v_i$$

**Reconstruction quality vs. compression ratio:**

| k components | Storage fraction | Typical visual quality |
|-------------|-----------------|----------------------|
| 1 | $\sim$0.2% | Recognisable shape, no detail |
| 10 | $\sim$2% | Major structures visible |
| 50 | $\sim$10% | Good — hard to distinguish from original at a glance |
| 100+ | $\sim$20%+ | Near-perfect reconstruction |

The module asks you to find the "elbow" in the reconstruction error vs. $k$ curve, where
adding more components gives diminishing returns.

### Latent space geometry

The low-dimensional projection is a **latent space** — a coordinate system where similar
images cluster together and smooth paths correspond to gradual image changes. Two
diagnostics to always run on a new latent space:

**2-D scatter plot:** project all images onto PC1 and PC2 and colour by class label or
metadata (disease vs. healthy, time point, staining protocol). If the classes are well
separated in the PCA scatter, a simple linear classifier will perform well. If they are
mixed, non-linear methods (t-SNE, UMAP, or a trained neural encoder) may be needed.

**Interpolation in latent space:** linearly interpolate between two images' projections and
reconstruct the intermediate points. A meaningful latent space produces smooth,
biologically plausible interpolations — intermediate vessel densities, gradual intensity
gradients. Abrupt or unrealistic transitions indicate that the projection has captured
noise or irrelevant variation.

### Connection to later phases

The latent-space intuition from this module reappears in:

- **Phase 02, Module 03** — the U-Net encoder compresses images into a bottleneck
  representation that is functionally a PCA-like latent space, but non-linear
- **Phase 04** — cryoEM 2-D class averaging compresses particle images into a small set
  of representative views; cryoDRGN (beyond this curriculum) encodes 3-D conformational
  heterogeneity in a continuous latent space learned by a VAE

---

## Notebook

| File | Description |
|------|-------------|
| `dimension_reduction_reconstruction.ipynb` | DRIVE dataset loading and pre-processing; per-pixel mean-subtraction; PCA via scikit-learn; explained variance plot and elbow detection; per-component image visualisation; reconstruction at k = 1, 5, 10, 50, 100; PSNR and SSIM reconstruction quality curves; 2-D latent scatter plot coloured by vessel density; latent interpolation demo |

---

## Dataset

**DRIVE** (Digital Retinal Images for Vessel Extraction) — 40 retinal fundus images with
manual vessel segmentation labels from two human graders, widely used as a benchmark for
vessel segmentation. In this module the images are used as a source of realistic biological
image patches for PCA experiments; the labels are used only to colour the scatter plot by
vessel density class. Download is handled automatically in the notebook's first cell.

---

## Setup

```bash
pip install scikit-learn numpy matplotlib pillow
```

No GPU is required. PCA on image patches completes in seconds on CPU.

---

## Common errors and how to fix them

| Error | Likely cause | Fix |
|-------|-------------|-----|
| First PC captures only brightness | Images were not mean-subtracted | Subtract the dataset mean image before fitting PCA |
| `sklearn.decomposition.PCA` is slow | Using dense SVD on a large matrix | Set `svd_solver='randomized'` and `n_components=k` to use truncated SVD |
| Reconstruction looks blurry even at k=100 | High-frequency detail lives in components > 100 | Increase k or check that normalisation is correct |
| PCA scatter plot shows no class structure | Classes differ in ways PCA cannot capture linearly | Try t-SNE or UMAP; or try computing PCA on features rather than raw pixels |

---

## Tips for beginners

- Always plot the **cumulative explained variance** curve before deciding how many
  components to keep. The "elbow" — where the curve flattens — is a principled choice that
  avoids both underfitting (too few components) and overfitting (too many, including noise
  directions).
- Visualise the **top 10 principal component images** before anything else. If PC1 looks
  like a brightness gradient and PC2 looks like the main structural variation, your
  normalisation may not be removing global illumination. Try histogram equalisation first.
- PCA is a linear method. It will find the best linear projection, but biological images
  often vary in complex non-linear ways (cell shape changes, morphological transitions).
  If PCA scatter plots show overlapping classes, that is not a failure of PCA — it is
  diagnostic information that non-linear methods are needed.
- A reconstruction PSNR of 28–32 dB with 50 components is typical for retinal images at
  this resolution. Use this as a sanity check that your pipeline is working.
