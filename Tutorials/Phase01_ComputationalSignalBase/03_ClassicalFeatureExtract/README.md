# 03 — Classical Feature Extraction: Edges, Lines & Curves

## What you will learn

- The convolution formula and its parameters (kernel size, padding, stride) that also govern CNN layers
- Applying **Gaussian kernels** for smoothing before feature detection
- **Edge detection** with Sobel, Roberts, Prewitt, and Canny operators
- Implementing edge detection from scratch as an image gradient
- **Line and curve detection** using the Hough Transform
- Using OpenCV and scikit-image in a real bio-image processing workflow

## Notebooks

| File | Description |
|------|-------------|
| `conv_edge_detection.ipynb` | Convolution mechanics, Gaussian kernels, and a suite of edge-detection operators |
| `line_curve_detection.ipynb` | Hough-based detection of straight lines and circular structures |

## Key concepts

**Edge** — a location where image intensity changes sharply. Mathematically, a local maximum of the image gradient magnitude.

**Sobel / Prewitt / Roberts operators** — small derivative kernels that approximate the image gradient in X and Y. They differ in noise sensitivity and computational cost.

**Canny detector** — a multi-stage pipeline (Gaussian smoothing → gradient magnitude → non-maximum suppression → double thresholding) that produces thin, well-localised edges. The gold standard for many applications.

**Hough Transform** — converts the edge-detection problem into a voting problem in parameter space, making it robust to gaps and noise. The circular Hough Transform is widely used to detect cell nuclei.

## Why this matters for bio-imaging

Cell boundaries, membrane contours, and filamentous structures are all captured by edge and line detectors. Many segmentation pipelines use these classical operators as a first pass before applying deep learning refinement.

## Setup

```bash
pip install opencv-python scikit-image numpy matplotlib
```
