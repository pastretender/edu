# 03 — Image Segmentation

## What you will learn

### Classical (geometric) methods
- **Threshold-based segmentation**: curve fitting to intensity histograms, Gaussian Mixture Models (GMM)
- **Region-growing**: seeded expansion for multi-channel images using confidence intervals
- **Watershed segmentation**: treating the image as a topographic surface and flooding from markers

### Deep learning methods
- **U-Net architecture**: encoder–decoder with skip connections, why it works for biomedical images with limited training data
- Using pre-built U-Net implementations from `segmentation_models_pytorch`, MONAI, and PyTorch Hub
- Instance segmentation on a real yeast-cell microscopy dataset

## Notebooks

| File | Description |
|------|-------------|
| `segmentation_geo_modified.ipynb` | Classical segmentation: thresholding, GMM, region-growing, watershed |
| `segmentation_deep.ipynb` | Deep U-Net segmentation on the yeast-cell instance segmentation dataset |

## Key concepts

**Semantic vs. instance segmentation** — semantic segmentation assigns a class label to every pixel; instance segmentation additionally distinguishes between individual objects of the same class (e.g., cell 1 vs. cell 2).

**U-Net** — originally designed for electron microscopy cell segmentation (Ronneberger et al., 2015). Its skip connections preserve spatial detail that would otherwise be lost in the bottleneck.

**Watershed** — models the image as a landscape where bright pixel values are peaks. Water "floods" from seed markers until regions meet, creating natural boundaries. Works well when objects touch.

**GMM (Gaussian Mixture Model)** — fits the pixel intensity histogram as a sum of Gaussians, each representing a tissue/region class. The threshold between classes is determined automatically from the fitted model.

## Datasets

- **Yeast-cell microscopy** — 493 densely annotated images with pixel-wise instance labels for yeast cells in microstructure traps ([arXiv:2304.07597](https://arxiv.org/abs/2304.07597))
- Classical notebooks use sample images included in scikit-image

## Setup

```bash
pip install scikit-image opencv-python segmentation-models-pytorch monai torch matplotlib
```
