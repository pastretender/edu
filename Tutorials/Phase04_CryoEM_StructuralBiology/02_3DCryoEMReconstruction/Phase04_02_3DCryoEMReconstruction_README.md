# 02 — 3-D CryoEM Reconstruction

## Overview

This module builds the complete 3-D reconstruction pipeline from first principles. Starting
from a known 3-D phantom, you will simulate realistic projection images (with CTF and
noise), correct them, and reconstruct the original 3-D structure using back-projection.
Then you will tackle the hardest problem in cryoEM: what do you do when the particle
orientations are completely unknown? The answer is expectation-maximisation, and you will
implement a simplified version from scratch.

---

## What you will learn

### The forward imaging model

Under the **weak-phase object approximation (WPOA)**, a thin biological specimen at low
electron dose behaves as a pure phase object. The image recorded on the detector is:

$$I(\mathbf{x}) = \text{CTF} * P_R[V](\mathbf{x}) + \eta(\mathbf{x})$$

where:
- $V(\mathbf{r})$ is the 3-D electron density of the molecule
- $P_R[V]$ is the 2-D projection of $V$ along direction $R$ (a rotation matrix)
- $*$ denotes convolution with the CTF point spread function
- $\eta$ is additive Gaussian + Poisson noise

The full forward model in code:
```
3D phantom V
    │
    ▼ project (sum along z after rotating by R)
2D projection P_R[V]
    │
    ▼ multiply by CTF in Fourier space
CTF-corrupted projection
    │
    ▼ add Gaussian noise + optional Poisson
Simulated micrograph image
```

### CTF correction

Without CTF correction, different particles acquired at different defoci will have their
CTF zeros at different spatial frequencies. When you average these particles, the zeros
will not align — some particles reinforce a given frequency, others cancel it. The net
result is a blurred, artefact-contaminated reconstruction.

**Phase-flip correction:**
$$\hat{I}_{\text{corrected}}(\mathbf{k}) = \text{sign}(\text{CTF}(\mathbf{k})) \cdot \hat{I}(\mathbf{k})$$
Restores phase coherence but leaves amplitude at CTF zeros unreliable.

**Wiener filter CTF correction:**
$$\hat{I}_{\text{corrected}}(\mathbf{k}) = \frac{\text{CTF}(\mathbf{k})}{\text{CTF}^2(\mathbf{k}) + \varepsilon} \cdot \hat{I}(\mathbf{k})$$
Optimal linear correction; requires estimating $\varepsilon$ (the inverse SNR at each
frequency). The notebook shows how changing $\varepsilon$ trades off noise amplification
near zeros against residual CTF artefacts.

### The Fourier Slice Theorem

The central theorem of tomographic reconstruction:

$$\boxed{\mathcal{F}_2\{P_R[\rho]\}(\mathbf{k}_\perp) = \mathcal{F}_3\{\rho\}(R\,\mathbf{k}_\perp)}$$

**In words:** the 2-D Fourier transform of a projection of a 3-D object is equal to the
values of the 3-D Fourier transform of that object along the central slice perpendicular
to the projection direction.

**Implication for reconstruction:** if you have projections from many different orientations
$R_1, R_2, \ldots, R_N$, you can fill in the 3-D Fourier volume slice by slice, then take
the inverse 3-D Fourier transform to recover the structure. This is the basis of all
Fourier reconstruction methods.

The notebook makes this tangible: watch the 3-D Fourier volume fill up as you add more
projections. The reconstructed density sharpens progressively as the coverage increases,
and you can see exactly which regions of Fourier space remain empty.

### Trilinear gridding back-projection

Inserting Fourier slices into a 3-D volume requires **interpolation** because the slice
values land at non-grid positions in 3-D Fourier space. Nearest-neighbour interpolation
causes severe artefacts. The standard approach is **trilinear gridding**:

1. For each Fourier pixel on the 2-D slice, distribute its value to the 8 surrounding 3-D
   grid points weighted by trilinear interpolation coefficients
2. Accumulate a **gridding weight volume** (the denominator)
3. Divide the accumulated signal volume by the weight volume (excluding points with zero
   weight)
4. Inverse 3-D Fourier transform

The gridding weight correction compensates for non-uniform sampling density — regions of
Fourier space covered by many projections would otherwise be over-represented.

### B-factor sharpening

The contrast transfer function, detector modulation transfer function, and particle motion
together impose an envelope function that damps high spatial frequencies. Even after CTF
correction, the reconstruction shows reduced amplitude at high resolution. **B-factor
sharpening** compensates by multiplying the Fourier amplitudes by an exponential:

$$F_{\text{sharp}}(\mathbf{k}) = F(\mathbf{k}) \cdot e^{B |\mathbf{k}|^2 / 4}$$

where $B$ is negative (amplifying high frequencies). The optimal $B$ is found via
**Guinier analysis**: plot $\ln(|F(\mathbf{k})|^2)$ against $|\mathbf{k}|^2$ — the slope
of the linear region is $B/4$.

### The missing wedge

In single-axis tilt tomography, the specimen can only be tilted to about ±70° (limited by
the geometry of the holder in the microscope column). This leaves a wedge-shaped region
of Fourier space unsampled — the **missing wedge**.

Consequences:
- Structures elongated along the beam axis (the Z axis, perpendicular to the tilt axis)
  are compressed by the missing frequencies — the characteristic Z-elongation artefact
- Template matching is less reliable in the beam direction
- Segmentation of membranes parallel to the beam axis is more error-prone

The notebook shows both the Fourier-space view (the missing wedge as an empty cone) and
the real-space view (a reconstructed sphere that is elongated along Z).

### 2-D class averaging before reconstruction

Before attempting 3-D reconstruction, production pipelines (RELION, cryoSPARC) always
run **2-D classification** on the particle stack. This serves three purposes:

1. **Junk removal** — false picks (ice crystals, carbon edges, aggregates) form diffuse
   or featureless 2-D classes and are discarded
2. **Data validation** — clearly defined 2-D class averages with visible structural
   features confirm the dataset is high quality
3. **Pre-denoising** — the class averages themselves are denoised versions of single-
   particle images, ready for 3-D reference generation

The module implements a simplified version (K-means + NCC alignment) to build intuition
before the EM reconstruction section.

### Expectation-Maximisation for pose estimation

In a real cryoEM experiment, the orientation of each particle on the cryo-grid is
**completely unknown**. You cannot solve this by brute-force search — a 3-D rotation has
3 continuous degrees of freedom (Euler angles). The standard solution is the **EM algorithm**,
which treats orientations as **latent variables**:

**E-step** (Expectation): For each particle image $I_i$, compute the posterior probability
$w_{ij}$ over a discrete set of reference orientations $\{R_j\}$:

$$w_{ij} \propto \exp\!\left(-\frac{\|I_i - \text{CTF}_i * P_{R_j}[V]\|^2}{2\sigma^2}\right)$$

This is proportional to the likelihood of observing particle $i$ given orientation $R_j$.

**M-step** (Maximisation): Update the 3-D volume as the weighted back-projection:

$$V^{(\text{new})}(\mathbf{r}) = \frac{\sum_{i,j} w_{ij}\, \text{BP}_{R_j}[I_i](\mathbf{r})}{\sum_{i,j} w_{ij}\, \text{BP}_{R_j}[1](\mathbf{r})}$$

Iterate until convergence. In practice, the algorithm requires a good initial reference
(often from a 2-D class average or a map from a homologous structure).

### Neural rendering / NeRF-style reconstruction

A recent alternative represents the 3-D density as an **implicit neural field**:
$f_\theta(x, y, z) \to \text{density}$. The entire forward model (project → CTF → noise)
is implemented as differentiable PyTorch operations, and the network weights $\theta$ are
optimised end-to-end by back-propagating through the imaging model.

This approach (related to cryoDRGN and 3DFlex) naturally handles **heterogeneous** datasets
where different particles represent different conformational states of the same molecule —
a classical single-volume EM reconstruction would produce a blurred average, while the
neural field can represent the full continuum.

---

## Notebooks

| File | Description |
|------|-------------|
| `cryoem_reconstruction_tutorial.ipynb` | 3-D phantom construction; full forward model (CTF + dose weighting + noise); CTF correction (phase-flip and Wiener); 2-D class averaging; Fourier Slice Theorem visualisation; trilinear-gridding back-projection; B-sharpening; SSNR and Guinier analysis; EM pose estimation loop; NeRF-style differentiable reconstruction |

---

## How this module connects to Module 01

Module 01 established the 2-D signal processing tools. Module 02 uses those tools in their
3-D context. Specifically:
- The CTF formula introduced in Module 01 is now applied to 3-D data
- The NCC alignment from Module 01 particle picking reappears in 2-D class averaging and
  in the EM E-step cross-correlation
- The FRC resolution concept from Module 01 extends to the **3-D FSC (Fourier Shell
  Correlation)** used in Module 02's SSNR evaluation

---

## Setup

```bash
pip install numpy scipy matplotlib scikit-image torch
```

No GPU is strictly required. The NeRF section runs considerably faster on GPU, but the
notebook falls back to CPU automatically.

---

## Tips for beginners

- Work through the **Fourier Slice Theorem demo** interactively before the back-projection
  section. Seeing the 3-D Fourier volume fill up slice by slice is the single most
  effective way to build intuition for how the algorithm works.
- Print `V.shape` and the shape of each intermediate at every pipeline stage. The forward
  model progresses through shapes `(N, N, N)` → `(N, N)` → `(N, N)` (complex) → `(N, N)`
  (real) — tracking these prevents silent shape bugs.
- The EM loop converges slowly for the first 5–10 iterations, then accelerates. Let it run
  to at least 20 iterations before evaluating the result.
- Negative Jacobian determinant values in the EM deformation output indicate a numerical
  problem, not a physical one — it means the step size is too large. Reduce `learning_rate`
  or `sigma`.
