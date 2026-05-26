# Phase 01 — Computational Signal Base

**Start here if you are new to image processing or PyTorch.**

This phase gives you the vocabulary and the toolkit that every later phase assumes. You will go from "what is a tensor?" to being able to convolve images, apply Fourier-domain filters, detect edges, and classify biological images using hand-crafted geometric features.

---

## Contents

| Subfolder | Topic | Notebooks |
|-----------|-------|-----------|
| [`01_PyTorchTensorBasic`](01_PyTorchTensorBasic/) | PyTorch tensors & basic image I/O | `introduction_to_pytorch.ipynb`, `basic_operations.ipynb` |
| [`02_FrequencyFiltering`](02_FrequencyFiltering/) | Spatial & Fourier-domain filtering | `image_filtering.ipynb`, `image_filtering_notebook.ipynb`, `Image_Pyramids_and_Frequency_Domain.ipynb` |
| [`03_ClassicalFeatureExtract`](03_ClassicalFeatureExtract/) | Convolution, edge & line/curve detection | `conv_edge_detection.ipynb`, `line_curve_detection.ipynb` |
| [`04_GeometryBasedClassification`](04_GeometryBasedClassification/) | Texture & geometry features for cell classification | `geometry_based_classification.ipynb` |

---

## Learning goals

After finishing this phase you will be able to:

- Create, reshape, and move tensors between CPU and GPU in PyTorch
- Load DICOM and TIFF medical images and inspect their metadata
- Apply smoothing, sharpening, and edge-detection filters in both the spatial and frequency domains
- Understand how the convolution operation underpins neural network layers
- Extract GLCM texture and geometric features from bio-images and feed them into a classical classifier

---

## Prerequisites

- Python & NumPy (basic comfort)
- No prior PyTorch or image-processing experience required

---

## Suggested order

`01_PyTorchTensorBasic` → `02_FrequencyFiltering` → `03_ClassicalFeatureExtract` → `04_GeometryBasedClassification`
