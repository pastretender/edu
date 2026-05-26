# 04 — Geometry-Based Classification of Bio-Images

## Overview

You have loaded images, filtered them, and detected edges. Now the question becomes: given
a segmented object — a cell, an organelle, a nucleus — how do you describe it numerically
so that a computer can tell one type from another?

This module answers that question using hand-crafted features: mathematical descriptors
that quantify an object's texture, shape, and morphology without any labelled training
data for the features themselves. These classical methods were the state of the art in
computational biology until roughly 2014. They remain valuable today in small-data
regimes, interpretable pipelines, and any context where a lightweight, reproducible
feature vector is needed.

---

## What you will learn

### The feature extraction pipeline

The full pipeline for geometry-based classification is:

```
Raw fluorescence image
        │
        ▼ Segmentation (thresholding or watershed — see Phase 02 Module 03)
Binary cell/organelle mask + intensity image
        │
        ├─► GLCM texture features (from intensity image within mask)
        │
        └─► Morphological/geometric features (from binary mask)
        │
        ▼ Feature vector per object  →  Classifier (SVM, k-NN, Random Forest)  →  Class label
```

This module focuses on the middle two steps — texture and geometry feature extraction —
assuming segmentation masks are already available.

### Texture features: the Gray-Level Co-occurrence Matrix (GLCM)

**Texture** is how pixel intensities are distributed spatially within a region. A smooth
region has slowly varying intensities; a granular texture has rapidly alternating bright
and dark pixels; a directional texture varies more in one direction than another.

The **Gray-Level Co-occurrence Matrix (GLCM)** captures texture by asking: how often does
a pixel with intensity $i$ appear adjacent to a pixel with intensity $j$, at a given
offset $(d, \theta)$?

$$\text{GLCM}[i, j] = \text{count of } \{(x, y) : f(x,y)=i,\; f(x+d\cos\theta,\, y+d\sin\theta)=j\}$$

The raw GLCM is a 2-D histogram. From it, four key scalar features are extracted:

| Feature | Formula (simplified) | Physical meaning |
|---------|---------------------|-----------------|
| **Contrast** | $\sum_{i,j}(i-j)^2 \cdot P[i,j]$ | Measures intensity variation; high for granular textures |
| **Correlation** | $\sum_{i,j}\frac{(i-\mu_i)(j-\mu_j)}{\sigma_i\sigma_j} P[i,j]$ | Measures linear dependence; high for directional/fibrous textures |
| **Energy** | $\sum_{i,j}P[i,j]^2$ | Measures uniformity; high for smooth or periodic textures |
| **Homogeneity** | $\sum_{i,j}\frac{P[i,j]}{1+|i-j|}$ | Measures closeness to diagonal; high when neighbours have similar intensity |

These four numbers capture surprisingly discriminative texture information without any
training. In the HeLa dataset used in this module, GLCM features alone achieve over 70%
classification accuracy across 10 organelle classes.

**Practical considerations:**

- Quantise the image to 8 or 16 grey levels before computing the GLCM (the matrix would
  be 256×256 at full 8-bit depth — unnecessarily large and slow)
- Compute the GLCM at multiple offsets $(d, \theta)$ and average to get rotation-invariant
  features
- Always normalise the GLCM (divide by the sum of all entries) before extracting features,
  so results are comparable across image sizes

### Morphological and geometric features

While texture features describe the intensity pattern inside an object, morphological
features describe the object's shape from its binary mask. These are computed by
`skimage.measure.regionprops`:

| Feature | Description | Biological interpretation |
|---------|-------------|--------------------------|
| **Area** | Pixel count | Cell size; organelle volume proxy |
| **Perimeter** | Boundary pixel count | Membrane length |
| **Eccentricity** | Ratio of focal distance to major axis length; 0=circle, 1=line | Elongation; mitotic cells become more round |
| **Solidity** | Area / convex hull area | Membrane ruffling; pseudopod extension |
| **Major / minor axis length** | Principal axes of the equivalent ellipse | Aspect ratio; orientation |
| **Euler number** | Number of objects minus number of holes | Topology; fragmented vs intact organelle |
| **Convex hull area** | Area of the smallest convex shape enclosing the object | Combined with Area gives solidity |

Morphological features are inherently scale-dependent. Normalise by cell area (or
equivalent circle diameter) before comparing across images from different magnifications.

### Mahotas: fast bio-image feature library

**Mahotas** is a C++-backed Python library optimised for texture and morphological feature
extraction from biological images. It provides two sets of features particularly useful
here:

- **Haralick features** — the GLCM-derived features above, computed efficiently in C++
  across all 13 directions in 3-D (or 4 in 2-D) simultaneously
- **Zernike moments** — rotation-invariant shape descriptors derived from orthogonal
  polynomials on the unit disk; capture higher-order shape information that simple
  geometric features miss

```python
import mahotas
features = mahotas.features.haralick(img_quantised).mean(axis=0)  # 13 Haralick features, averaged over directions
```

### The HeLa cell organelle dataset

The **HeLa dataset** (Boland & Murphy, 2001; widely used as a benchmark since) contains
fluorescence microscopy images of HeLa cells labelled with antibodies targeting ten
different subcellular compartments:

| Class | Organelle |
|-------|-----------|
| 0 | Endosome |
| 1 | Lysosome |
| 2 | Mitochondria |
| 3 | Nucleus |
| 4 | Endoplasmic reticulum |
| 5 | Actin |
| 6 | Golgi apparatus |
| 7 | Plasma membrane |
| 8 | Microtubule |
| 9 | Nucleolus |

The classification challenge: many organelles have overlapping texture and morphology —
mitochondria and ER are both fibrous; nucleus and nucleolus are both roughly circular.
GLCM + geometric features distinguish them well enough to demonstrate the approach; deep
learning (Phase 02) closes the remaining gap with learned representations.

### Classical classifiers for image features

Once features are extracted, any tabular classification algorithm can be applied. The
module covers:

**SVM (Support Vector Machine):** finds the maximum-margin hyperplane separating classes
in feature space. With an RBF kernel, handles non-linear boundaries. Use
`sklearn.svm.SVC(kernel='rbf', C=1.0, gamma='scale')`. Robust to high-dimensional feature
vectors when $n_{\text{samples}} \gg n_{\text{features}}$.

**k-NN (k-Nearest Neighbours):** classifies a sample by majority vote among its $k$ nearest
neighbours in feature space. No training phase — just memorise the training set. Simple
and interpretable but slow at inference for large datasets. Sensitive to feature scaling;
always normalise before k-NN.

**Feature normalisation:** always apply `sklearn.preprocessing.StandardScaler` before
any distance-based classifier (k-NN, SVM with RBF). GLCM contrast values can be in the
thousands while eccentricity is between 0 and 1; without normalisation, large-valued
features dominate the distance computation.

### When geometry-based features still win

Even in the deep-learning era, hand-crafted features have genuine advantages:

- **Small data:** a feature extractor trained on 50 images can match a CNN trained on
  50,000. If you have fewer than a few hundred labelled examples, start here.
- **Interpretability:** a researcher can inspect the feature importance and say "cells
  classified as mitotic have higher eccentricity and lower solidity." A CNN bottleneck
  vector cannot be explained the same way.
- **Speed:** extracting GLCM features takes milliseconds; running a U-Net takes seconds.
  For large-scale screening experiments with millions of cells, this matters.
- **Downstream statistics:** if you need to run a statistical test comparing feature
  distributions across experimental conditions, a named feature vector is the right input.

---

## Notebook

| File | Description |
|------|-------------|
| `geometry_based_classification.ipynb` | Full pipeline: HeLa dataset download and loading; GLCM quantisation and computation at multiple offsets; Haralick feature extraction with Mahotas; morphological feature extraction with skimage.regionprops; feature matrix construction; StandardScaler normalisation; SVM and k-NN classification; confusion matrix and per-class accuracy; feature importance analysis |

---

## Setup

```bash
pip install mahotas scikit-image scikit-learn numpy matplotlib
```

If `from skimage.feature import graycomatrix` raises an `ImportError`, upgrade scikit-image:
```bash
pip install --upgrade scikit-image
```

No GPU is required. All computation is CPU-based and completes in under 5 minutes for the
full HeLa dataset.

---

## Common errors and how to fix them

| Error | Likely cause | Fix |
|-------|-------------|-----|
| Mahotas `haralick` returns NaN | Image has all-zero intensity (empty mask region) | Add a guard: skip objects where `img.sum() == 0` |
| SVM takes very long to fit | Large dataset without normalisation (gradient descent on huge scale differences) | Apply `StandardScaler` before fitting; use `SVC(kernel='rbf', max_iter=1000)` |
| Low accuracy despite many features | Features correlated; classifier overfitting | Use `sklearn.decomposition.PCA` to reduce dimensionality before classification |
| `regionprops` returns no objects | Binary mask is all zeros | Check segmentation step; verify threshold produces foreground pixels |
| `graycomatrix` function not found | Older scikit-image API | Use `skimage.feature.graycomatrix` (not `greycomatrix`); upgrade if needed |

---

## Tips for beginners

- Before fitting any classifier, visualise the **feature distributions** as histograms or
  box plots grouped by class. If the distributions overlap heavily, no classifier will
  perform well on those features — either the features are not discriminative or the
  segmentation masks are noisy.
- Use `sklearn.model_selection.cross_val_score` with 5-fold cross-validation rather than a
  single train/test split. With small datasets (a few hundred samples), a single split has
  high variance.
- The GLCM must be computed on the **quantised** image. Use
  `(img / img.max() * 15).astype(np.uint8)` to quantise to 16 levels. Computing the GLCM
  on raw 16-bit pixel values produces a 65536 × 65536 matrix — far too large.
- Plot the confusion matrix. Overall accuracy hides per-class imbalance: a classifier
  that always predicts the most common class looks acceptable on accuracy but fails
  completely on rare classes.
