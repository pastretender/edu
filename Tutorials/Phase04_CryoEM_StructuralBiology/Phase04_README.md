# Phase 04 — CryoEM & Structural Biology

**Level: Specialist** · Phase 01 Module 02 (Fourier transforms) and NumPy/SciPy fluency are required. No prior structural-biology background is assumed — the physics is developed from first principles.

CryoEM is computationally demanding because the experiment cannot provide clean data. To prevent radiation damage the electron dose must be kept extremely low — roughly 20–60 electrons per square ångström — which means the raw micrographs have SNR between 0.01 and 0.1. The signal is buried 10–100× beneath noise. Recovering a protein structure from these images requires a tightly coupled chain of signal-processing steps, each of which this phase builds from the ground up. You will simulate the imaging physics yourself, implement the reconstruction mathematics in code, and finally apply the same ideas to a real published dataset using the production software stack.

---

## Contents

| Subfolder | Topic | Files |
|-----------|-------|-------|
| [`01_LowSNRNoiseProcessing`](01_LowSNRNoiseProcessing/) | SNR intuition, CTF simulation, classical denoising, particle picking, 2-D class averaging, FRC, Noise2Void | `cryoem_low_snr_tutorial.ipynb` |
| [`02_3DCryoEMReconstruction`](02_3DCryoEMReconstruction/) | Forward model, CTF correction, Fourier Slice Theorem, trilinear back-projection, EM pose estimation, NeRF-style reconstruction | `cryoem_reconstruction_tutorial.ipynb` |
| [`03_TomogramDiagnosisBuild`](03_TomogramDiagnosisBuild/) | Hands-on cryo-ET workflow on EMPIAR-10164: tilt-series stacking, fiducial alignment, WBP and SIRT reconstruction, tomogram diagnosis | `build_and_diagnose_tomogram.md` |

Modules 01 and 02 are CPU-only Jupyter notebooks and run on Google Colab. Module 03 is a local software practical requiring IMOD/Etomo on Linux or macOS.

---

## Module Summaries

### Module 01 — Low-SNR Image Processing for CryoEM

Everything in cryoEM starts from one uncomfortable fact: no single image of a biological molecule is interpretable. The electron dose required to preserve the specimen keeps SNR so low that signal and noise are indistinguishable in any individual frame. The solution is coherent averaging — collect thousands to millions of copies of the same molecule, align them, and average. Noise cancels stochastically while signal accumulates, improving SNR as √N. This module makes that insight concrete by measuring it directly: watch SNR improve from −10 dB to +10 dB as the number of averaged particles grows from 1 to 1000.

The central complication is the **Contrast Transfer Function (CTF)** — the oscillating phase-contrast function imposed by the microscope's optics. The CTF destroys information completely at specific spatial frequencies (visible as Thon rings in the power spectrum), and images acquired at different defoci have their zeros at different positions. Particles cannot be averaged coherently without first correcting this phase modulation. The module implements and compares phase-flip correction (fast, restores phase coherence but not amplitude at zeros) and Wiener filtering (optimal under Gaussian noise, requires a noise PSD estimate). Classical denoising methods — Gaussian smoothing, median filtering, and Butterworth bandpass filtering — are benchmarked with PSNR and SSIM; the bandpass filter is shown to be the standard pre-processing choice because it cleanly separates ice background gradients, particle signal, and detector read-out noise.

The module then builds the two downstream tools that feed into Module 02: template-matching particle picking (normalised cross-correlation in Fourier space, with a threshold sweep showing the recall-precision trade-off) and 2-D class averaging (K-means clustering of NCC-aligned particle boxes, with SNR measured directly). Fourier Ring Correlation (FRC) is introduced as the gold-standard resolution measurement, and the module closes with a conceptual introduction to Noise2Void — the leading self-supervised deep denoiser for cryoEM, where no clean ground-truth images exist.

**Key skills:** CTF formula and defocus effects, `np.fft.fft2` for Fourier filtering, NCC particle picking in Fourier space, K-means class averaging, FRC and the 0.143 half-bit criterion, Noise2Void blind-spot concept.

---

### Module 02 — 3-D CryoEM Reconstruction

Module 01 established 2-D micrograph-level signal processing. Module 02 extends every concept into 3-D and builds the complete reconstruction pipeline from mathematical first principles. The starting point is the forward imaging model under the **weak-phase object approximation**: a 3-D electron density $V(\mathbf{r})$ is projected to 2-D along a rotation $R$, convolved with the CTF point spread function, and corrupted with noise. All three operations are implemented as differentiable PyTorch code, establishing the foundation for the NeRF-style reconstruction section at the end.

The **Fourier Slice Theorem** is the central result: the 2-D Fourier transform of a projection equals the values of the 3-D Fourier transform along the central slice perpendicular to the projection direction. The module makes this tangible by visualising the 3-D Fourier volume filling up slice by slice as more projections are added at different angles — the single most effective way to build reconstruction intuition. **Trilinear gridding back-projection** implements this correctly: because Fourier slices land at non-grid positions in 3-D, values must be distributed to neighbouring grid points with trilinear interpolation weights, and a density-compensation weight volume corrects for non-uniform Fourier-space coverage.

The hardest real-world problem in cryoEM — that particle orientations are completely unknown on the cryo-grid — is addressed by **Expectation-Maximisation pose estimation**. The E-step computes posterior weights over a discrete set of reference orientations by cross-correlation; the M-step updates the 3-D volume as a weighted back-projection. A simplified version is implemented from scratch and shown to converge over 20+ iterations. The module closes with a **NeRF-style differentiable reconstruction** that represents 3-D density as an implicit neural field optimised end-to-end through the full forward model — an approach that naturally handles conformational heterogeneity that a single-volume EM reconstruction would smear.

**Key skills:** weak-phase object approximation, CTF Wiener correction, Fourier Slice Theorem proof and code, trilinear gridding, B-sharpening and Guinier analysis, missing-wedge visualisation, EM algorithm E/M steps, `torch.nn.functional.grid_sample` for differentiable warp.

**Connects to:** Module 03 — every concept (CTF, missing wedge, WBP, SIRT, NCC alignment, FSC) reappears as production software settings in Etomo, giving you the vocabulary to interpret every dialogue box.

---

### Module 03 — Build & Diagnose a Real Cryo-ET Tomogram

This is a hands-on practical rather than a Jupyter notebook. You will process a real cryo-ET tilt series of *Mycoplasma pneumoniae* from the EMPIAR-10164 public archive using IMOD/Etomo — the same software stack used by structural-biology labs worldwide. The three processing steps (stack → align → reconstruct) are carried out in full, and four guided diagnostic questions connect every result back to the computational theory in Modules 01 and 02.

**Step A — Tilt-stack assembly:** Each tilt angle is recorded as a movie of multiple frames that must be motion-corrected and averaged before stacking. With Warp available, its per-patch polynomial motion model produces the best results; the IMOD-only path using `alignframes` also works with slightly lower quality. The output is a single `.st` file with one corrected image per tilt angle.

**Step B — Fiducial-based alignment:** Colloidal gold beads are tracked through the full tilt range in Etomo to fit a per-tilt geometric transformation. The target is a mean fiducial residual below 0.5 pixels; high-tilt images routinely contribute the worst per-tilt residuals due to foreshortened bead geometry and longer beam paths through the ice.

**Step C — Dual reconstruction:** Both Weighted Back Projection (WBP — fast, linear, gold standard for sub-tomogram averaging) and SIRT (iterative, smoother, less streaking, preferred for segmentation) are reconstructed and compared side by side in 3dmod.

Four diagnostic questions require written answers and 3dmod screenshots: alignment quality (fiducial residuals), the missing-wedge artefact (Z-elongation of spherical objects), WBP vs SIRT (streaking, sharpness, application suitability), and pixel size/Nyquist limit.

**Key skills:** IMOD/Etomo fiducial tracking, residual interpretation, WBP radial filter, SIRT iteration count, 3dmod Slicer tool, missing-wedge diagnosis, pixel size and Nyquist calculation.

---

## Learning Goals

After finishing this phase you will be able to:

- Explain why cryoEM micrographs have SNR < 0.1 and derive the √N SNR improvement from coherent averaging
- Write the CTF formula, simulate its effect in code, and correct it with phase-flipping and Wiener filtering
- Perform template-matching particle picking via normalised cross-correlation and evaluate the recall-precision trade-off as a function of detection threshold
- Compute 2-D class averages and read a Fourier Ring Correlation resolution curve; explain why the 0.143 half-bit criterion is the correct threshold
- Derive the Fourier Slice Theorem and implement trilinear-gridding back-projection with density compensation
- Understand how the EM algorithm jointly estimates particle orientations and the 3-D reconstruction, and implement a simplified version from scratch
- Use IMOD/Etomo to align a real cryo-ET tilt series with gold fiducials and reconstruct both WBP and SIRT tomograms
- Identify the missing-wedge artefact in a real tomogram and explain its origin in Fourier space

---

## Prerequisites

- **Phase 01, Module 02** (Fourier transforms, spatial frequencies, FFT) — used throughout all three modules
- NumPy, SciPy, and Matplotlib fluency
- **Module 03 only:** IMOD/Etomo installed on a local Linux or macOS machine; approximately 20 GB free disk space; internet access for the EMPIAR dataset download

---

## Hardware Requirements

| Module | Compute | Notes |
|--------|---------|-------|
| 01 — Low-SNR Processing | CPU only | All NumPy/SciPy; Colab free tier is sufficient |
| 02 — 3-D Reconstruction | CPU (most sections), GPU optional | The NeRF section benefits from GPU but falls back to CPU |
| 03 — Tomogram Practical | Local machine with IMOD | Colab-incompatible; Linux or macOS required (WSL2 on Windows) |

---

## Setup

```bash
# Modules 01 and 02 — Python dependencies
# Each notebook also installs its own requirements in the first cell
pip install numpy scipy matplotlib scikit-image torch tqdm

# Module 03 — IMOD (Linux, Ubuntu 20.04+)
# Download the installer from https://bio3d.colorado.edu/imod/download.html
sudo bash imod_4.XX.XX_RHEL7-64_CUDA10.1.sh
source /etc/imod-setup.sh   # add to ~/.bashrc for persistence
3dmod --version             # verify installation

# Warp (optional, recommended for Module 03 motion correction)
# https://www.warpem.com/warp/
```

**Start the EMPIAR-10164 download before anything else.** TS_01 is 8–12 GB and takes 30–60 minutes on a typical connection. Follow the teamtomo preparation script: https://teamtomo.org/teamtomo-site-archive/walkthroughs/EMPIAR-10164/preparation.html

---

## Suggested Order

```
01_LowSNRNoiseProcessing
         ↓
02_3DCryoEMReconstruction
         ↓
03_TomogramDiagnosisBuild
```

The order is strict. Module 03 exposes you to real data and production software; every Etomo setting and every result it produces maps directly to a concept introduced computationally in Modules 01 and 02. Starting Module 03 without that background makes the software difficult to interpret and the diagnostic questions impossible to answer from principle.

---

## How the Modules Connect

Module 01 builds the 2-D micrograph-level signal processing: why images are so noisy, how CTF corrupts specific spatial frequencies, and how averaging recovers the signal. Module 02 takes each of those ideas into 3-D: the Fourier Slice Theorem shows how to assemble a 3-D structure from the projections Module 01 produced, and EM-based pose estimation handles the real experimental problem of unknown particle orientations. Module 03 applies the complete pipeline on real data using production software — every dialogue box in Etomo corresponds to a parameter you set in code in Modules 01 or 02.

| Computational concept (Modules 01–02) | Production equivalent (Module 03) |
|--------------------------------------|-----------------------------------|
| CTF simulation and phase-flip correction | CTF correction toggle in Etomo before reconstruction |
| Particle picking by NCC | Fiducial bead tracking across the tilt series |
| Missing wedge in synthetic back-projection | Missing-wedge artefact in the real tomogram (Q2) |
| WBP via Fourier Slice Theorem | Etomo `radial` filter WBP reconstruction |
| SIRT iterative loop (conceptual) | Etomo SIRT with configurable iteration count |
| FRC resolution curve | Visual assessment and Nyquist calculation (Q4) |

---

## Further Reading

- Kühlbrandt, W. (2014). "The resolution revolution." *Science* 343, 1443–1444.
- Nogales, E. & Scheres, S.H.W. (2015). "Cryo-EM: A Unique Tool for the Visualization of Macromolecular Complexity." *Molecular Cell* 58, 677–689.
- Grant, T. & Grigorieff, N. (2015). "Measuring the optimal exposure for single particle cryo-EM using a 2.6 Å reconstruction of rotavirus VP6." *eLife* 4, e06980.
- Mastronarde, D.N. (1997). "Dual-axis tomography: an approach with alignment methods that preserve resolution." *Journal of Structural Biology* 120, 343–352.
- RELION tutorial: https://relion.readthedocs.io/en/release-5.0/
- cryoSPARC guide: https://guide.cryosparc.com/
- teamtomo EMPIAR-10164 walkthrough: https://teamtomo.org
- IMOD/Etomo user guide: https://bio3d.colorado.edu/imod/doc/etomoTutorial.html
