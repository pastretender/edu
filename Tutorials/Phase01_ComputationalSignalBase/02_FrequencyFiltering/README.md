# 02 — Frequency Filtering & Image Pyramids

## What you will learn

- The mathematical definition of 2-D convolution and why every filter is just a weighted neighbourhood average
- **Spatial-domain filters**: Box (mean), Gaussian blur, Median, Laplacian sharpening
- **Frequency-domain thinking**: what the Fourier Transform reveals about an image and why some operations are far faster in frequency space
- Low-pass, high-pass, and band-pass filtering in the Fourier domain
- **Image pyramids**: building Gaussian and Laplacian pyramids for multi-scale analysis
- Trade-offs between each method and when to prefer one over another

## Notebooks

| File | Description |
|------|-------------|
| `image_filtering.ipynb` | Comprehensive walkthrough of every major filter type with side-by-side visual comparisons |
| `image_filtering_notebook.ipynb` | Companion exercises and additional examples |
| `Image_Pyramids_and_Frequency_Domain.ipynb` | Gaussian & Laplacian pyramids; blending and reconstruction |

## Key concepts

**Convolution** — sliding a small kernel across an image and computing the dot product at every position. The choice of kernel determines the filter's effect (blur, sharpen, edge-detect, etc.).

**Fourier Transform** — decomposes an image into its constituent spatial frequencies. Low frequencies = coarse structure; high frequencies = fine detail and noise.

**Image pyramid** — a sequence of progressively downsampled (and sometimes band-pass filtered) versions of an image. Essential for scale-invariant detection and image blending.

## Practical context

Before any machine-learning model ever sees a microscopy or medical image, one or more of these filters has almost certainly been applied — to reduce noise, equalise contrast, or extract multi-scale features. Understanding them prevents "black-box" surprises later.

## Setup

```bash
pip install numpy scipy scikit-image matplotlib opencv-python
```
