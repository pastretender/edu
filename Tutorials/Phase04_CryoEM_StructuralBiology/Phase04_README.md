# Phase 04 — CryoEM & Structural Biology

**Level: Specialist** · Phase 01 Module 02 (Fourier transforms) and NumPy/SciPy fluency are required. No prior structural-biology background is assumed — the relevant physics is developed from first principles.

CryoEM is one of the most computationally demanding fields in modern science. The electron dose must remain so low to prevent radiation damage that typical micrograph SNR is between 0.01 and 0.1 — signal buried 10 to 100 times below noise. Recovering atomic-resolution protein structure from these images requires a set of signal-processing techniques that are among the most sophisticated in applied science. This phase builds the foundational understanding that makes those techniques interpretable, from the forward imaging model all the way to a real cryo-ET tomogram processed with production software.

---

## The Resolution Revolution

CryoEM underwent a step-change around 2013 driven by three simultaneous advances:

- **Direct electron detectors** replacing scintillator cameras — dramatically lower readout noise, enabling movie-mode acquisition for beam-induced motion correction
- **CTF correction** methods that had previously been approximate became accurate to sub-ångström precision
- **Maximum-likelihood reconstruction algorithms** (RELION, then cryoSPARC) replacing older back-projection with full Bayesian inference over particle orientations

The result: structures that had previously required crystallography or been simply inaccessible (membrane proteins, large dynamic complexes) could now be solved at 2–4 Å resolution. Understanding the computational machinery behind this revolution is the goal of Phase 04.

---

## Contents

| # | Subfolder | Core topic | Files | GPU / local install? |
|---|-----------|-----------|-------|----------------------|
| 1 | [`01_LowSNRNoiseProcessing`](01_LowSNRNoiseProcessing/) | CTF simulation, denoising, particle picking, class averaging, FRC | `cryoem_low_snr_tutorial.ipynb` | No (CPU only) |
| 2 | [`02_3DCryoEMReconstruction`](02_3DCryoEMReconstruction/) | Forward model, CTF correction, Fourier Slice Theorem, back-projection, EM pose estimation, NeRF reconstruction | `cryoem_reconstruction_tutorial.ipynb` | Optional (NeRF section) |
| 3 | [`03_TomogramDiagnosisBuild`](03_TomogramDiagnosisBuild/) | Hands-on cryo-ET workflow on EMPIAR-10164: stacking, fiducial alignment, WBP/SIRT reconstruction, tomogram diagnosis | `build_and_diagnose_tomogram.md` | Yes — IMOD installed locally |

---

## Module Summaries

### Module 01 — Low-SNR Image Processing for CryoEM

The entry point to cryoEM computation. Everything starts from a single central insight: each electron micrograph of a biological molecule is so noisy that no single image is interpretable. The solution is averaging — collect hundreds of thousands of copies of the same molecule, align them, and average them. Noise cancels; signal adds coherently; SNR improves as √N.

This module makes that insight concrete by simulating a complete cryoEM micrograph from scratch: start with a clean phantom, apply the Contrast Transfer Function (CTF) that the microscope's optics impose, add Poisson shot noise and Gaussian readout noise, and arrive at a realistic low-SNR image. Every subsequent step is a systematic attempt to recover the original phantom.

**CTF** is the central complication. The oscillating phase-contrast function destroys information completely at specific spatial frequencies (Thon rings). Images acquired at different defoci have their zeros in different places — particles cannot be averaged directly without first restoring phase coherence. The module implements and compares phase-flip correction (fast, does not restore amplitude) and Wiener filtering (optimal under Gaussian noise, requires noise PSD estimate).

**Classical denoising** — Gaussian smoothing, median filtering, Wiener filtering, and bandpass filtering — are implemented and benchmarked with PSNR and SSIM. The bandpass filter (Butterworth high-pass × low-pass) is shown to be the standard choice because it cleanly separates the three frequency regimes: ice background gradients, particle signal, and detector noise.

**Particle picking** via normalised cross-correlation (NCC) is implemented in Fourier space for efficiency. Local maxima above a confidence threshold give particle coordinates; the threshold sweep demonstrates the recall-precision trade-off explicitly.

**2-D class averaging** extracts particle boxes, aligns them by maximising NCC over in-plane rotations and translations, clusters with K-means, and averages within each class. The SNR improvement from averaging 1000 particles per class is measured directly.

**Fourier Ring Correlation (FRC)** is introduced as the gold-standard resolution measurement: split the particle stack in two, average each half independently, and measure the Fourier correlation as a function of spatial frequency. The frequency at which correlation drops below 0.143 defines resolution. The module demonstrates why visual smoothness is a misleading proxy for true resolution — a blurry average can look nicer than a grainier one that actually passes FRC at higher resolution.

**Noise2Void** is introduced conceptually as the leading self-supervised deep-denoising approach for cryoEM, where no clean ground-truth exists.

**Key skills:** CTF formula and parameter effects, `np.fft.fft2` for Fourier filtering, NCC in Fourier space, K-means class averaging, FRC curve, `noise2void` blind-spot concept.

---

### Module 02 — 3-D CryoEM Reconstruction

Module 01 established 2-D signal processing. Module 02 extends everything to 3-D and builds the complete reconstruction pipeline from first principles.

**The forward imaging model** is derived under the weak-phase object approximation: a 3-D molecule projected to 2-D, convolved with the CTF point spread function, and corrupted with noise. All three operations are implemented as differentiable code, establishing the foundation for the NeRF-style section at the end.

**The Fourier Slice Theorem** is the central theorem of tomographic reconstruction: the 2-D Fourier transform of a projection equals the values of the 3-D Fourier transform along the central slice perpendicular to the projection direction. The module visualises the 3-D Fourier volume filling up slice by slice as more projections are added from different angles — the single most effective way to build intuition for how reconstruction works.

**Trilinear gridding back-projection** implements the reconstruction algorithm correctly: Fourier slices land at non-grid positions in 3-D, so values must be distributed to neighbouring grid points with trilinear interpolation weights. A weight volume (the density of Fourier coverage) divides the accumulated signal to correct for non-uniform sampling.

**B-factor sharpening** compensates for the exponential amplitude fall-off at high frequencies caused by CTF envelope, particle motion, and detector modulation. Guinier analysis identifies the optimal B from the linear slope of the log-power spectrum.

**The missing wedge** — the cone of Fourier space unsampled in single-axis tilt tomography (which cannot tilt beyond ~±70°) — is shown in both Fourier space (the missing cone) and real space (a reconstructed sphere elongated along the beam axis). The practical consequences for downstream analysis are discussed.

**Expectation-Maximisation pose estimation** tackles the hardest real-world problem: particle orientations are completely unknown on the cryo-grid. The EM algorithm treats orientations as latent variables: the E-step computes posterior weights over a discrete set of reference orientations via cross-correlation; the M-step updates the 3-D volume as a weighted back-projection. The module implements a simplified version and shows convergence over 20+ iterations.

**NeRF-style differentiable reconstruction** represents the 3-D density as an implicit neural field and optimises it end-to-end by back-propagating through the full forward model. This approach handles heterogeneous datasets (multiple conformational states) where classical single-volume EM would produce a blurred average.

**Key skills:** weak-phase object approximation, CTF phase-flip and Wiener correction, Fourier Slice Theorem, trilinear gridding, B-sharpening, Guinier analysis, missing wedge, EM algorithm latent-variable formulation, `torch.nn.functional.grid_sample` for differentiable warp.

---

### Module 03 — Build & Diagnose a Real Cryo-ET Tomogram (EMPIAR-10164)

A hands-on practical rather than a Jupyter notebook. You will process real cryo-ET data of *Mycoplasma pneumoniae* from the EMPIAR public archive using IMOD/Etomo — the same software stack used by structural-biology labs worldwide.

The practical bridges the gap between "I understand the algorithm in a notebook" and "I can operate production software on real data."

**Step A — Tilt-series stacking:** Each tilt angle is recorded as a movie of multiple frames that must be aligned and averaged to correct beam-induced sample motion. The corrected frames are stacked into a single file that Etomo can process. With Warp available: follow the teamtomo motion-correction workflow. Without Warp: use IMOD's `alignframes` directly.

**Step B — Fiducial alignment:** Gold bead fiducials are tracked automatically through the full tilt range. The goal is a mean fiducial residual below 0.5 pixels — larger residuals indicate the tracked bead model does not fit the data well and the reconstruction will be blurred. High-tilt images (>±50°) typically contribute the largest per-tilt residuals because foreshortening elongates the fiducials and makes tracking harder.

**Step C — Dual reconstruction:** Reconstruct with Weighted Back Projection (WBP, fast, linear, standard for sub-tomogram averaging) and SIRT (iterative, suppresses streaking from high-contrast objects, better for visualisation and segmentation). Save both, compare in 3dmod.

**Four diagnostic questions** anchor the practical to the computational theory from Modules 01 and 02:

- **Q1 — Alignment quality:** record the final mean residual and identify which tilts contribute the worst residuals
- **Q2 — Missing wedge:** reconstruct with and without high-tilt views, compare Z-elongation of spherical objects (gold beads, ribosomes) in 3dmod's Slicer
- **Q3 — WBP vs SIRT:** compare streaking, sharpness, contrast, and suitability for template matching vs segmentation on the same region of interest
- **Q4 — Pixel size:** explain how detector pixel size, magnification, and binning combine to set the Nyquist limit, and assess whether the reconstruction approaches that limit visually

**Key skills:** IMOD/Etomo fiducial tracking, residual analysis, WBP `radial` weighting filter, SIRT iteration count choice, 3dmod Slicer tool, missing-wedge visualisation.

---

## Learning Goals

After completing Phase 04 you will be able to:

- Explain why cryoEM micrographs have SNR < 0.1 and derive the √N SNR improvement that averaging provides
- Write the CTF formula, explain the role of each parameter (defocus, spherical aberration, wavelength), simulate its effect in code, and correct it with both phase-flipping and Wiener filtering
- Perform template-matching particle picking via normalised cross-correlation and evaluate the recall-precision trade-off
- Compute 2-D class averages and read a Fourier Ring Correlation (FRC) resolution curve — and explain why the 0.143 half-bit criterion is the correct threshold
- Derive the Fourier Slice Theorem from scratch and implement trilinear-gridding back-projection
- Understand how the EM algorithm jointly estimates particle orientations and the 3-D reconstruction volume, and implement a simplified version
- Use IMOD/Etomo to align a real cryo-ET tilt series with fiducials and produce both WBP and SIRT reconstructions
- Identify the missing-wedge artefact in a real tomogram and explain its origin in Fourier space

---

## How the Modules Build on Each Other

```
Module 01 — 2-D signal processing           Module 02 — 3-D extension
──────────────────────────────────────       ────────────────────────────────────
Simulate CTF + noise              ──────►    Same CTF, applied to 3-D projections
Classical denoising (bandpass)    ──────►    Pre-processing before reconstruction
Particle picking (NCC)            ──────►    NCC reused in EM E-step
2-D class averaging               ──────►    2-D averages initialise 3-D refinement
FRC resolution                    ──────►    Extends to 3-D FSC (Fourier Shell Correlation)
Noise2Void concept                           NeRF-style differentiable reconstruction

                    Module 03 — Real data
                    ─────────────────────────────────────
                    All concepts above in production software
                    IMOD fiducial alignment
                    WBP vs SIRT reconstruction comparison
                    Tomogram diagnosis on EMPIAR-10164
```

---

## Prerequisites

- **Phase 01, Module 02** (Fourier transforms, spatial frequency, FFT) — used constantly throughout
- NumPy, SciPy, and Matplotlib fluency
- **Module 03 only:** IMOD/Etomo installed on a local Linux or macOS machine (not Colab-compatible); ~20 GB free disk space for the dataset download

---

## Hardware Requirements

| Module | Compute | Notes |
|--------|---------|-------|
| 01 — Low-SNR Processing | CPU only | All NumPy/SciPy; Colab CPU runtime sufficient |
| 02 — 3-D Reconstruction | CPU (most), GPU optional | NeRF section benefits from GPU but falls back to CPU |
| 03 — Tomogram Practical | Local machine with IMOD | Linux or macOS; WSL2 on Windows |

---

## Software Installation (Module 03)

```bash
# IMOD — Linux (Ubuntu 20.04+)
# Download from https://bio3d.colorado.edu/imod/download.html and run:
sudo bash imod_4.XX.XX_RHEL7-64_CUDA10.1.sh

# Warp (optional, recommended for motion correction)
# https://www.warpem.com/warp/

# Add IMOD to your PATH
source /etc/imod-setup.sh   # add to ~/.bashrc for persistence
```

---

## Dataset Download (Module 03)

The practical uses tilt series TS_01 from EMPIAR-10164. Follow the teamtomo preparation script to download the per-tilt multi-frame `.mrc` files and SerialEM `.mdoc` metadata:

```
https://teamtomo.org/teamtomo-site-archive/walkthroughs/EMPIAR-10164/preparation.html
```

Approximate download size for TS_01: **8–12 GB**. Begin this download well before you plan to start the practical.

---

## Suggested Order

```
01_LowSNRNoiseProcessing
         ↓
02_3DCryoEMReconstruction
         ↓
03_TomogramDiagnosisBuild    ← requires Modules 01 + 02 to interpret the software output
```

The ordering is important. Module 03 will present you with IMOD dialogue boxes and reconstruction outputs that will be difficult to interpret without the CTF, missing wedge, WBP, and SIRT background from Modules 01 and 02.

---

## Further Reading

- Kühlbrandt, W. (2014). "The resolution revolution." *Science* 343, 1443–1444. — Short accessible overview of why cryoEM became dominant after 2013.
- Nogales, E. & Scheres, S.H.W. (2015). "Cryo-EM: A Unique Tool for the Visualization of Macromolecular Complexity." *Molecular Cell* 58, 677–689.
- Grant, T. & Grigorieff, N. (2015). "Measuring the optimal exposure for single particle cryo-EM using a 2.6 Å reconstruction of rotavirus VP6." *eLife* 4, e06980. — The dose-weighting framework behind optimal exposure estimation.
- RELION tutorial: https://relion.readthedocs.io/en/release-5.0/
- cryoSPARC guide: https://guide.cryosparc.com/
- teamtomo walkthrough (used in Module 03): https://teamtomo.org
- IMOD/Etomo user guide: https://bio3d.colorado.edu/imod/doc/etomoTutorial.html
