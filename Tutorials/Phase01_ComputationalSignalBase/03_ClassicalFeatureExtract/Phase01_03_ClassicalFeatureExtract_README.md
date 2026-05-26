# 03 — Classical Feature Extraction: Edges, Lines & Curves

## Overview

Before a neural network can segment a cell or detect a structure, someone decided what
features matter. For decades the answer was edges, lines, and curves — the boundaries and
filaments that define the geometry of biological structures. These classical detectors are
still widely used today: as pre-processing steps before deep learning, as ground-truth
generators for evaluation, and as the building blocks of many domain-specific pipelines.

This module also makes the connection between hand-crafted convolution kernels and the
learned kernels inside a convolutional neural network explicit — by the end, a CNN's first
layer will look less like a black box and more like an automatic version of what you will
build by hand here.

---

## What you will learn

### Convolution mechanics: the operation behind CNN layers

The convolution formula was introduced in Module 02. This module unpacks its parameters in
the context of feature detection, because these are the same parameters you configure in
`torch.nn.Conv2d`:

| Parameter | Meaning | Typical values |
|-----------|---------|---------------|
| `kernel_size` | Spatial extent of the filter | 3, 5, 7 |
| `padding` | Pixels added to the border before convolving | 0 (no padding), 1 (for 3×3 to preserve size) |
| `stride` | Step size between filter applications | 1 (full resolution), 2 (halves output size) |
| `dilation` | Spacing between kernel elements (dilated convolution) | 1 (standard), 2 (doubles receptive field without extra params) |

A 3×3 Conv2d layer with 64 output channels is, structurally, identical to applying 64
different hand-crafted 3×3 kernels to the image — except the kernels are learned from data
rather than designed by a human.

### Gaussian pre-smoothing

Before detecting edges, noise must be suppressed. A noise spike produces a large gradient
that looks identical to a real edge. The standard approach is to convolve the image with a
Gaussian kernel first:

$$f_\sigma = G_\sigma * f$$

then detect edges in $f_\sigma$. The choice of $\sigma$ sets a minimum scale: features
smaller than $\sigma$ pixels are suppressed before detection. This scale selection is
critical — too small and noise edges dominate; too large and fine structural details are
blurred away before they can be found.

### Edge detection operators

An **edge** is a location where image intensity changes rapidly. The image gradient
$\nabla f = [\partial f/\partial x,\; \partial f/\partial y]$ is large at edges. Each
detector approximates this gradient differently:

**Sobel operator:** uses a 3×3 kernel that weights the central row/column more heavily,
giving it better noise robustness than simpler finite differences:

$$K_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \qquad
K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}$$

Edge magnitude: $\|\nabla f\| = \sqrt{(f*K_x)^2 + (f*K_y)^2}$

**Prewitt operator:** same structure as Sobel but without the central weighting; slightly
less noise-robust but simpler to analyse mathematically.

**Roberts operator:** a 2×2 cross-derivative operator — the smallest possible gradient
approximation. Fast but highly noise-sensitive; rarely used in practice today.

**Comparison:**

| Operator | Kernel size | Noise robustness | Localisation accuracy |
|----------|-------------|------------------|-----------------------|
| Roberts | 2×2 | Poor | Good |
| Prewitt | 3×3 | Moderate | Moderate |
| Sobel | 3×3 | Good | Moderate |
| Canny | Multi-stage | Excellent | Excellent |

### Canny edge detector: the gold standard

Canny (1986) is not a single convolution kernel but a four-stage pipeline designed to
satisfy three formal criteria: low error rate, good localisation, and single-response
(each edge produces exactly one detected edge, not a thick band).

**Stage 1 — Gaussian smoothing:**
$f_\sigma = G_\sigma * f$

**Stage 2 — Gradient magnitude and direction:**
$M = \sqrt{G_x^2 + G_y^2}$, $\theta = \arctan(G_y / G_x)$

**Stage 3 — Non-maximum suppression (NMS):**
For each pixel, suppress it (set to 0) if it is not a local maximum along the gradient
direction $\theta$. This thins edges from thick bands to single-pixel lines.

**Stage 4 — Double thresholding and hysteresis:**
Strong pixels (above high threshold) are definite edges. Weak pixels (between low and high
thresholds) are kept only if they are connected to a strong pixel. Pixels below the low
threshold are discarded. This eliminates isolated noise responses while preserving edge
continuity.

**Tunable parameters:** $\sigma$ (smoothing scale), `low_threshold`, `high_threshold`. The
ratio `low:high` is typically 1:2 or 1:3.

### Hough Transform: detecting lines and circles

After edge detection, you have a binary image of edge pixels. The **Hough Transform** asks:
"which of these pixels are part of the same line or circle?" It does so by mapping each
edge pixel to a curve of *all lines that pass through it* in a parameter space, then
finding where many such curves intersect (= a line many pixels agree on).

**Standard Hough Transform for lines** uses the normal parameterisation $\rho = x\cos\theta + y\sin\theta$:

1. For each edge pixel $(x, y)$: vote for all $(\rho, \theta)$ pairs satisfying the equation
2. Accumulate votes in a 2-D histogram (the Hough accumulator)
3. Local maxima in the accumulator correspond to lines in the image

This formulation handles vertical lines (which would require infinite slope in y = mx + b).

**Circular Hough Transform:** for circles of known radius $r$, each edge pixel $(x, y)$
votes for a ring of possible centres at distance $r$. The accumulator is a 2-D image of
centre votes. For unknown radius, the accumulator becomes 3-D (centre + radius).

**Why Hough is powerful in bio-imaging:** cell nuclei are approximately circular; filaments
are approximately straight. The Hough Transform finds these structures robustly even when
they are partly occluded or incomplete — as long as enough pixels vote for the same
parameter, the structure is detected.

**In practice:** `skimage.transform.hough_line_peaks` and `skimage.transform.hough_circle`
implement these efficiently. The main tuning decisions are the accumulator resolution and
the minimum peak height.

### The connection to CNN layers

This table will make Phase 02's deep-learning modules feel familiar:

| Classical concept | CNN equivalent |
|------------------|---------------|
| Sobel X kernel | A learned 3×3 Conv2d filter at layer 1 (early layers typically learn edge-like filters) |
| Gaussian pre-smoothing | Implicit regularisation via batch normalisation and weight decay |
| Multi-scale filtering | Pooling + skip connections in U-Net |
| Hough voting | Attention mechanism: each pixel votes for relevant spatial relationships |

The key difference: classical kernels are designed by humans based on mathematical
analysis; CNN kernels are optimised by gradient descent on labelled data. But they solve
the same underlying problem.

---

## Notebooks

| File | Description |
|------|-------------|
| `conv_edge_detection.ipynb` | Convolution parameters (kernel_size, padding, stride, dilation); Gaussian pre-smoothing with σ sweep; Sobel, Prewitt, and Roberts gradient computation; Canny edge detection with threshold sweep; PSNR/F1 comparison of edge methods on a ground-truth cell boundary dataset |
| `line_curve_detection.ipynb` | Standard Hough Transform for straight lines on a synthetic grid and a real microscopy image; Circular Hough Transform for nucleus detection; parameter sensitivity analysis (accumulator resolution, peak threshold) |

---

## Why this matters for bio-imaging

Cell and tissue morphology is fundamentally geometric. Every downstream quantitative
analysis — measuring cell elongation, counting mitotic figures, tracing axon branches,
quantifying collagen fibre orientation — starts with detecting the relevant geometric
primitives. Classical edge and line detectors provide the first pass; deep-learning
refinement (Phase 02, Module 03) handles the ambiguous cases.

Even in pipelines that use U-Net for segmentation, Canny edges or Sobel gradients often
appear as additional input channels — providing the model with pre-computed local gradient
information that would otherwise require many training examples to learn.

---

## Setup

```bash
pip install opencv-python scikit-image numpy matplotlib
```

No GPU is required.

---

## Common errors and how to fix them

| Error | Likely cause | Fix |
|-------|-------------|-----|
| Canny output is nearly all white | Thresholds too low for the image's intensity range | Normalise image to [0, 1] first, then set thresholds in [0.05, 0.3] |
| Canny output is nearly all black | Thresholds too high | Lower `low_threshold`; check the image has been converted to float |
| Hough lines detect noise, not real lines | Edge image too noisy | Apply stricter Canny thresholds or increase Gaussian σ before Hough |
| Circular Hough misses small nuclei | Radius range does not cover the actual size | Measure a few nuclei manually first; set `min_radius` and `max_radius` accordingly |
| `cv2.Canny` expects uint8 | Image is float32 | Convert with `(img * 255).astype(np.uint8)` before calling |

---

## Tips for beginners

- Always visualise intermediate stages: Gaussian-smoothed image → gradient magnitude →
  NMS → final edges. Skipping straight to the final output makes it impossible to diagnose
  which stage is causing problems.
- Plot the gradient magnitude image before thresholding. It shows you the full distribution
  of edge strengths, which makes setting `low_threshold` and `high_threshold` for Canny
  straightforward.
- For the Hough Transform, a dense accumulator (small bin size) gives better precision but
  takes longer; a coarse accumulator (large bin size) is fast but misses closely-spaced
  parallel lines. Start with the default and refine.
- The Canny detector in OpenCV (`cv2.Canny`) expects uint8 input and thresholds in [0,
  255]. The scikit-image version (`skimage.feature.canny`) expects float in [0, 1] and
  thresholds in the same range. They are not interchangeable without rescaling.
