# Phase 04 — CryoEM & Structural Biology

**Specialist level.** This phase is aimed at students and researchers entering the cryoEM
or cryo-ET field. A working knowledge of Fourier analysis (Phase 01, Module 02), basic
image filtering, and Python scientific computing (NumPy / SciPy) is assumed. No prior
structural-biology background is required — the physics is developed from first principles.

---

## Why CryoEM is computationally demanding

Cryo-electron microscopy images biological molecules embedded in vitreous ice and
illuminated by a focused electron beam. The electron dose must remain extremely low to
prevent radiation damage to the specimen, which means only a handful of electrons per
square ångström reach the detector. The resulting images have **SNR between 0.01 and 0.1
— signal buried 10–100× below noise**. Recovering protein structure from these images
requires signal-processing pipelines that are among the most sophisticated in all of
applied science.

The field has undergone a **resolution revolution** since ~2013, driven by:
- Direct electron detectors replacing photographic film and scintillator-based cameras
- More accurate CTF correction methods
- Maximum-likelihood and Bayesian reconstruction algorithms (RELION, cryoSPARC)
- Deep-learning tools for denoising, particle picking, and model building

This phase builds the foundational computational understanding that makes those tools
interpretable, from the forward imaging model all the way to a real tomogram.

---

## Contents

| Subfolder | Topic | Files |
|-----------|-------|-------|
| [`01_LowSNRNoiseProcessing`](01_LowSNRNoiseProcessing/) | SNR intuition, CTF simulation, classical denoising, particle picking, class averaging, FRC, and Noise2Void | `cryoem_low_snr_tutorial.ipynb` |
| [`02_3DCryoEMReconstruction`](02_3DCryoEMReconstruction/) | Forward model, CTF correction, Fourier Slice Theorem, trilinear back-projection, EM pose estimation, and NeRF-style reconstruction | `cryoem_reconstruction_tutorial.ipynb` |
| [`03_TomogramDiagnosisBuild`](03_TomogramDiagnosisBuild/) | Hands-on cryo-ET workflow using IMOD/Etomo on real EMPIAR-10164 data: tilt-series stacking, fiducial alignment, WBP/SIRT reconstruction, and tomogram diagnosis | `build_and_diagnose_tomogram.md` |

---

## How the modules build on each other

Module 01 establishes the 2-D micrograph-level signal processing: why images look so
noisy, how CTF corrupts specific spatial frequencies, and how averaging many copies of the
same molecule recovers the signal. Module 02 extends these ideas into 3-D: the Fourier
Slice Theorem shows how to assemble a 3-D structure from the 2-D projections produced in
Module 01, and EM-based pose estimation deals with the real problem of not knowing the
orientation of each particle. Module 03 takes everything into the lab: you apply the same
concepts using production software on a real published dataset.

```
Module 01                        Module 02                    Module 03
─────────────────────────────    ─────────────────────────    ──────────────────────────
Simulate CTF + noise             Forward model (3D phantom     Real data (EMPIAR-10164)
Classical denoising              → projections → CTF → noise)  Tilt-series stacking
Particle picking (NCC)           CTF correction                Fiducial alignment (Etomo)
2D class averaging               Fourier Slice Theorem         WBP reconstruction
FRC resolution                   Back-projection               SIRT reconstruction
Noise2Void concept               EM pose estimation            Tomogram diagnosis
                                 NeRF reconstruction
```

---

## Learning goals

After finishing this phase you will be able to:

- Explain why cryoEM micrographs have SNR < 0.1 and derive the √N SNR improvement from averaging
- Write the CTF formula, simulate its effect in code, and correct it with phase-flipping and Wiener filtering
- Perform template-matching particle picking via normalised cross-correlation
- Compute 2-D class averages and read a Fourier Ring Correlation (FRC) curve
- Derive the Fourier Slice Theorem and implement trilinear back-projection from scratch
- Understand how the EM algorithm jointly estimates particle orientations and the 3-D reconstruction
- Use IMOD/Etomo to align a real cryo-ET tilt series and reconstruct both WBP and SIRT tomograms
- Identify the missing-wedge artefact in a real tomogram and explain its Fourier-space origin

---

## Prerequisites

- **Phase 01, Module 02 (Frequency Filtering)** — Fourier transforms are used throughout
- NumPy, SciPy, and Matplotlib fluency
- Module 03 requires **IMOD/Etomo installed locally** (not Colab-compatible)

---

## Hardware

Modules 01 and 02 run on CPU and are Colab-compatible (no GPU required). Module 03 is a
local software practical requiring IMOD on Linux or macOS.

---

## Suggested order

`01_LowSNRNoiseProcessing` → `02_3DCryoEMReconstruction` → `03_TomogramDiagnosisBuild`

The order is important here. Module 03 requires understanding CTF, missing wedge, WBP,
and SIRT — all introduced computationally in Modules 01 and 02. Starting with Module 03
without the background will make the IMOD dialogue boxes difficult to interpret.

---

## Further reading

- Kühlbrandt, W. (2014). "The resolution revolution." *Science* 343, 1443–1444.
  — A short, accessible overview of why cryoEM became dominant after 2013.
- Nogales, E. & Scheres, S.H.W. (2015). "Cryo-EM: A Unique Tool for the Visualization of
  Macromolecular Complexity." *Molecular Cell* 58, 677–689.
- RELION tutorial: https://relion.readthedocs.io/en/release-5.0/
- cryoSPARC guide: https://guide.cryosparc.com/
- teamtomo walkthrough (used in Module 03): https://teamtomo.org
