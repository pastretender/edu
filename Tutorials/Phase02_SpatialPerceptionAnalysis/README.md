# Phase 02 — Spatial Perception & Analysis

**Intermediate level.** You should be comfortable with PyTorch tensors and basic filtering (Phase 01) before starting here.

This phase is about understanding *where* things are in an image — reducing high-dimensional data to a meaningful low-dimensional space, detecting and tracking objects, delineating their boundaries, and aligning images that were captured at different times or with different instruments.

---

## Contents

| Subfolder | Topic | Notebooks |
|-----------|-------|-----------|
| [`01_DimensionReductionLatentSpace`](01_DimensionReductionLatentSpace/) | PCA & latent-space methods for bio-image analysis | `dimension_reduction_reconstruction.ipynb` |
| [`02_ObjectDetectTracking`](02_ObjectDetectTracking/) | YOLOv5 object detection + centroid-based 3-D tracking | `object_detection.ipynb`, `LAP_LapTrack.ipynb` |
| [`03_ImageSegmentation`](03_ImageSegmentation/) | Classical (threshold, watershed) and deep (U-Net) segmentation | `segmentation_geo_modified.ipynb`, `segmentation_deep.ipynb` |
| [`04_ImageAlignmentRegister`](04_ImageAlignmentRegister/) | Classical registration (SimpleITK) + image-to-image translation (Pix2Pix) | `Image_Registration.ipynb`, `image_translation.ipynb` |

---

## Learning goals

After finishing this phase you will be able to:

- Compress high-dimensional image data with PCA, reconstruct images from principal components, and interpret what each component represents
- Train a YOLOv5 detector on a custom dataset and evaluate its bounding-box predictions
- Track objects across 3-D time-lapse data by linking centroids with the LAP algorithm
- Segment cells and structures using threshold, Gaussian mixture, region-growing, and watershed methods
- Build and train a U-Net for instance segmentation
- Register a pair of images using gradient-descent optimisation over a similarity metric
- Train a Pix2Pix GAN to translate out-of-focus microscopy images into in-focus ones

---

## Prerequisites

- Phase 01 completed
- Basic familiarity with Python classes and training loops

---

## Suggested order

`01_DimensionReductionLatentSpace` → `03_ImageSegmentation` → `02_ObjectDetectTracking` → `04_ImageAlignmentRegister`
