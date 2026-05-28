# 03 — Build & Diagnose a Cryo-ET Tomogram

## Overview

This is a **hands-on practical** rather than a Jupyter notebook. You will process a real
cryo-ET tilt series from the public dataset **EMPIAR-10164** using IMOD/Etomo — the same
software stack used by structural-biology labs worldwide. Working through three processing
steps (stack → align → reconstruct) and four guided diagnostic questions will translate the
computational theory from Modules 01 and 02 into the muscle memory of operating production
software on experimental data.

The dataset is *Mycoplasma pneumoniae* imaged by plunge-freezing cryo-ET. The reconstructed
tomograms show the cytoplasmic membrane, densely packed ribosomes (~25 nm diameter), and
surface protein complexes including the attachment organelle — making it an excellent
learning dataset because there are visually obvious spherical and planar structures to use
as landmarks when evaluating reconstruction quality and missing-wedge artefacts.

By the end you will have produced and compared two tomographic reconstructions of the same
tilt series, identified the missing-wedge artefact in your own data, and made a quantitative
judgement about alignment quality, reconstruction method choice, and achievable resolution.
Every result you observe will map directly to a concept introduced computationally in
Modules 01 or 02; the table in the final section of this README makes those connections
explicit.

**Complete Modules 01 and 02 first.** The CTF, missing wedge, WBP, SIRT, and fiducial
residual concepts are all introduced computationally in those modules. Arriving without that
background makes the Etomo dialogue boxes difficult to interpret and the diagnostic questions
impossible to answer from principle rather than by guessing.

---

## What you will learn

### The cryo-ET acquisition geometry

CryoET imaging is not a single exposure. The specimen is mounted on a tilting stage and
imaged sequentially at discrete angles — a **tilt series** — to provide the angular coverage
needed for 3-D reconstruction. A typical acquisition covers −60° to +60° in 3° steps
(roughly 40 images), but the physical geometry of the cryo-holder limits maximum tilt to
approximately ±70°. The angular range that is never imaged is the **missing wedge**: a
cone-shaped region of Fourier space that remains permanently unsampled, causing
direction-dependent smearing artefacts in the final tomogram.

Each tilt is also recorded in **movie mode**: the detector reads out many successive frames
during continuous beam exposure. This allows post-hoc alignment of the frames to correct
**beam-induced specimen motion** — the tendency of the thin ice layer to shift and buckle
under electron irradiation. Without motion correction, each tilt image is a blurred
superposition of slightly shifted frames, degrading all features and reducing the achievable
resolution.

### Step A — Motion correction and tilt-stack assembly

The first processing step converts the raw per-tilt movie files into a single aligned stack
that Etomo can reconstruct. Two paths are available:

**With Warp (recommended):** Warp fits a per-frame, per-patch polynomial motion trajectory
to each patch of the detector independently, then averages the corrected frames to produce a
single motion-corrected image per tilt. It exports an IMOD-compatible `.st` stack and an
updated `.mdoc` metadata file. Follow the teamtomo "Stack generation" walkthrough for the
exact steps.

**IMOD-only path:** The IMOD command `alignframes` performs whole-frame alignment by
cross-correlating successive frames, then `newstack` assembles the corrected images into a
`.st` stack. The motion model is less sophisticated than Warp's patch-based approach, so
residual motion will be slightly higher — but the downstream alignment and reconstruction
pipeline is identical.

After this step you should have a single `.st` file with approximately 40 images ordered
from most negative to most positive tilt angle.

### Step B — Fiducial-based tilt-series alignment

The tilt stack is a sequence of 2-D images each recorded at a slightly different physical
stage position. Before reconstruction, all images must be brought into a common coordinate
frame. EMPIAR-10164 uses **colloidal gold fiducials** — spherical gold nanoparticles (~10 nm
diameter) adsorbed to the specimen surface that appear as bright, compact, high-contrast
dots and can be tracked reliably across the full tilt range.

The alignment model fitted by Etomo is a per-tilt geometric transformation:

$$\mathbf{x}'_i = s_i\, R(\phi_i)\, (\mathbf{x}_i - \mathbf{c}_i) + \mathbf{t}_i$$

where $s_i$ is a per-tilt magnification, $R(\phi_i)$ an in-plane rotation, $\mathbf{c}_i$
the tilt axis, and $\mathbf{t}_i$ a translation. The parameters are estimated by minimising
the sum of squared distances between the observed 2-D bead positions and those predicted
by projecting a fitted 3-D bead model through each transform. This residual error is the
primary quality indicator for the alignment step.

**Interpreting residuals:**

| Mean residual | Assessment |
|---------------|------------|
| < 0.5 px | Excellent — reconstruction will be close to the theoretical resolution limit |
| 0.5–1.0 px | Acceptable for most purposes |
| 1.0–1.5 px | Marginal — check for poorly tracked beads or large inter-tilt stage-shift jumps |
| > 1.5 px | Poor — tracking has failed for some tilts; investigate before reconstructing |

High-tilt images (> ±50°) routinely contribute the worst per-tilt residuals. At steep
angles the foreshortened geometry stretches the projected bead into an elongated ellipse,
the in-plane SNR drops (the beam path through the specimen is longer), and any imperfection
in the stage tilt is magnified in the image plane. This is expected behaviour — but a
per-tilt residual more than 3–4× the mean for any single image suggests that tilt is
corrupted or that the bead model failed there, and it can be excluded before reconstruction.

**Practical tracking workflow in Etomo:**
1. Set the gold bead radius precisely — measure several bead diameters in 3dmod and divide by 2
2. Run Automatic Seed Finding to detect beads in the zero-degree tilt
3. Run Beadtrack to propagate bead positions through the full tilt range
4. Inspect the bead model in 3dmod and manually correct any incorrectly tracked beads
5. Rerun the alignment optimisation until residuals stabilise

### Step C — Weighted Back Projection (WBP)

WBP is the direct implementation of the Fourier Slice Theorem introduced in Module 02: the
2-D Fourier transform of each CTF-corrected tilt image is inserted into the appropriate
plane of a 3-D Fourier volume, and the filled volume is inverse-transformed to produce the
reconstruction. In practice this is implemented in real space as back-projection, with a
**radial weighting filter** (the `radial` setting in Etomo) applied to each tilt before
accumulation.

The radial filter corrects for the fact that low-spatial-frequency regions of Fourier space
receive contributions from more projection directions than high-frequency regions. Without
compensation, back-projection over-weights low frequencies and the reconstruction is blurred.
The standard ramp filter $|k|$ multiplied by a low-pass roll-off compensates this density
difference and produces the characteristic high-contrast, high-noise WBP appearance.

| Property | WBP |
|----------|-----|
| Speed | Fast — a single pass through the tilt series |
| Noise level | High — the ramp filter amplifies all frequencies including noise |
| Streaking | Present near high-contrast objects (gold beads, carbon edges) |
| Intensity scale | Linear — density values are quantitatively comparable |
| Best for | Sub-tomogram averaging, template matching, FSC resolution measurement |

### Step C — Simultaneous Iterative Reconstruction Technique (SIRT)

SIRT is an iterative algorithm that applies no frequency-domain filter. Instead it starts
from a zero-filled volume and refines it by repeatedly forward-projecting the current
estimate at all tilt angles, computing the difference from the measured tilt images,
back-projecting the error, and updating the volume by adding a fraction of that error.
After 10–50 iterations the volume converges toward a solution that minimises the total
squared discrepancy between the measured tilts and the forward projections of the
reconstruction.

Because SIRT never applies a ramp filter, it does not amplify high-frequency noise, and
the iterative error-correction suppresses the linear streaking artefacts that dominate WBP
near high-contrast objects. The trade-off is that SIRT is substantially slower and produces
an intensity scale that is not directly proportional to electron density — making
quantitative density comparisons across volumes unreliable.

| Property | SIRT |
|----------|------|
| Speed | Slow — each iteration repeats the full forward and backward projection |
| Noise level | Low — iterative convergence implicitly suppresses high-frequency noise |
| Streaking | Greatly reduced compared to WBP |
| Intensity scale | Non-linear — values not directly comparable across volumes |
| Best for | Visual inspection, manual segmentation, automated segmentation |

### The missing wedge in a real tomogram

Module 02 simulated the missing wedge from synthetic projections. Here you will observe it
in your own data. In real space the effect is direction-dependent smearing: any feature
whose density extends along the beam axis (Z) is elongated in Z because the Fourier
components that would define its Z extent are missing. Gold beads are the cleanest
diagnostic — they are perfectly spherical in reality, so every deviation from a sphere in
3dmod is a pure reconstruction artefact.

To observe the missing wedge in 3dmod's Slicer:

```bash
3dmod TS_01_WBP.rec    # open the reconstruction
# Ctrl+T → Slicer window
# Set slab thickness to ~5 nm
# Middle-click drag to rotate around the Y axis (the tilt axis)
# Watch a gold bead elongate as you approach 90° (beam direction)
```

Reconstructing a second time after excluding all tilts beyond ±40° (Etomo's "Exclude Views")
exaggerates the missing wedge from ±20° to ±50°, making the Z-elongation dramatically more
visible and connecting directly to the Fourier-space cone shown in Module 02.

### Pixel size, binning, and the Nyquist limit

Every cryo-ET reconstruction has a physical pixel size that sets the resolution ceiling,
determined by the chain:

$$\text{pixel size} = \frac{\text{detector pixel size (µm)}}{\text{magnification}} \times \text{binning factor}$$

The **Nyquist frequency** is $1 / (2 \times \text{pixel size in Å})$, the highest spatial
frequency that can be represented. No structural information finer than this scale can be
recovered regardless of the reconstruction algorithm. **Binning** trades resolution for
speed and contrast: a bin-4 tomogram runs ~16× faster than bin-1 but has 4× worse Nyquist
limit. For EMPIAR-10164, bin 4–8 is typical for initial visual inspection and template
matching; final sub-tomogram averaging is done at bin 1 or 2.

### The four diagnostic questions

All four questions must be answered in `build_and_diagnose_tomogram.md`, each with a written
answer and a screenshot from Etomo or 3dmod.

**Q1 — Alignment quality:** Record the final mean fiducial residual from the Etomo alignment
log. Identify which tilt angles contribute the largest per-tilt residuals and give the
physical explanation for why those angles are worse. Expected target: < 0.5 px is good;
> 1.5 px indicates a tracking problem that should be resolved before reconstruction.

**Q2 — Missing-wedge artefact:** Produce two reconstructions — one with the full tilt range,
one excluding all tilts beyond ±40°. In 3dmod's Slicer, compare the shape of gold beads and
ribosome densities in both. Describe the artefact in Z vs XY, and explain why membranes
parallel to the XY plane appear normal while membranes parallel to the beam axis are smeared
or invisible.

**Q3 — WBP vs SIRT:** Select the same region of interest (a membrane patch, a cluster of
ribosomes, and a gold bead) in both reconstructions. Assess which method shows less linear
streaking, which looks sharper vs smoother, and which you would trust for template matching
vs for manual or automated segmentation. Justify each choice from the properties tables
above.

**Q4 — Pixel size and Nyquist limit:** From the Etomo log or the tomogram header, note the
pixel size in ångströms. Calculate the Nyquist limit. Assess visually whether the structural
features you can resolve approach that limit, and suggest whether changing the binning factor
would benefit a downstream sub-tomogram averaging project.

---

## Files

| File | Description |
|------|-------------|
| `build_and_diagnose_tomogram.md` | Step-by-step instructions and answer templates for all four diagnostic questions |

---

## Dataset

**EMPIAR-10164** — cryo-ET data of *Mycoplasma pneumoniae* cells prepared by plunge-freezing,
deposited in the Electron Microscopy Public Image Archive (EMPIAR), the community repository
for raw cryoEM data. The reconstructed tomograms show the cytoplasmic membrane, packed
ribosomes, and surface complexes including the attachment organelle, making it a visually
rich and widely used teaching dataset.

For this practical you will work with **TS_01** from a 5-tilt-series subset that the
teamtomo walkthrough provides a download script for:

```
https://teamtomo.org/teamtomo-site-archive/walkthroughs/EMPIAR-10164/preparation.html
```

The download consists of per-tilt multi-frame `.mrc` files and SerialEM `.mdoc` metadata
files encoding acquisition parameters, tilt angles, nominal defocus, and stage position
for each frame. **Start the download before anything else.** Total size for TS_01 is 8–12 GB
and may take 30–60 minutes on a typical connection.

---

## How this module connects to Modules 01 and 02

Modules 01 and 02 built the computational foundations in Python. Module 03 applies the same
concepts in production software on real data.

| Computational concept (Modules 01–02) | Production equivalent (Module 03) |
|--------------------------------------|-----------------------------------|
| CTF simulation and phase-flip correction | CTF correction toggle in Etomo before WBP reconstruction |
| Missing wedge in synthetic back-projection | Missing-wedge Z-elongation observed in 3dmod Slicer (Q2) |
| WBP via Fourier Slice Theorem | Etomo `radial` filter weighted back-projection |
| SIRT iterative error-correction loop | Etomo SIRT with configurable iteration count |
| NCC alignment of 2-D class averages | Fiducial bead tracking across the tilt series |
| FRC resolution curve | Nyquist limit calculation and visual assessment (Q4) |

---

## Setup

No Python environment is needed. All processing uses IMOD/Etomo.

```bash
# IMOD — Linux (Ubuntu 20.04+)
# Download the installer from https://bio3d.colorado.edu/imod/download.html
sudo bash imod_4.XX.XX_RHEL7-64_CUDA10.1.sh

# Make IMOD commands available in your current shell
source /etc/imod-setup.sh   # add this line to ~/.bashrc for persistence

# Verify the installation
3dmod --version
etomo --help
```

For Warp installation and configuration see https://www.warpem.com/warp/

---

## Common errors and how to fix them

| Error | Likely cause | Fix |
|-------|-------------|-----|
| Fiducial tracking fails to find seeds | Bead radius set too large or too small | Open one tilt in 3dmod, measure several bead diameters in pixels, divide by 2, re-enter in Etomo |
| Residuals do not converge below 2 px | Poor bead distribution; large stage-shift jump in `.mdoc` | Exclude tilts with anomalously large per-tilt residuals; add more manually seeded beads across the field of view |
| Reconstruction is entirely black | Wrong `.ali` file path or incorrect tilt axis orientation | Confirm the aligned stack `.ali` is non-empty (`ls -lh TS_01.ali`); check the tilt axis angle in Etomo |
| 3dmod crashes when opening the `.rec` | Reconstruction did not finish | Check that the Etomo log shows a normal exit; re-run if any error is present in the log |
| IMOD commands not found after installation | PATH not updated | Run `source /etc/imod-setup.sh` and add this line to `~/.bashrc` |
| Warp refuses to import `.mrc` files | Frame count mismatch with `.mdoc` | Ensure you downloaded all frame files for each tilt; the `.mdoc` line count should match the file count |

---

## Tips for beginners

- **Start the download before anything else.** TS_01 is 8–12 GB. Starting it the evening before the session saves a frustrating wait at the beginning.
- Open 3dmod and inspect the raw tilt stack before running any alignment. You should see gold beads clearly in the zero-degree tilt. If you cannot, something is wrong with the stack assembly step — troubleshoot there rather than in Etomo.
- When bead tracking fails repeatedly, do not keep re-running automatic tracking with the same settings. Instead, manually pick 5–10 beads spread across the field of view in the first, middle, and last tilt images, then let Beadtrack propagate from those seeds. Manual seed placement solves the majority of tracking failures.
- Compare WBP and SIRT at the **same Z slice** at the **same display contrast**. Use 3dmod's Image Processing → Contrast window to normalise both before comparing — WBP and SIRT have very different intensity ranges, and comparing them without normalisation is misleading.
- When reconstructing with the reduced ±40° tilt range for Q2, save it as a separate file (`TS_01_limited.rec`) rather than overwriting your full reconstruction.
- Gold beads are the most reliable diagnostic for Q2. They are perfectly spherical in reality, so every deviation you see in 3dmod is a pure reconstruction artefact with no ambiguity from biological shape variation.
- The Slicer is essential for Q2. Set the slab thickness to 1 voxel, orient it perpendicular to the tilt axis, and rotate around the Y axis while watching a single bead — the elongation appears and disappears as you rotate through the beam direction.

---

## Further reading

- Mastronarde, D.N. (1997). "Dual-axis tomography: an approach with alignment methods that preserve resolution." *Journal of Structural Biology* 120, 343–352. — The original paper introducing the fiducial alignment approach implemented in IMOD.
- Bharat, T.A.M. & Scheres, S.H.W. (2016). "Resolving macromolecular structures from electron cryo-tomography data using subtomogram averaging in RELION." *Nature Protocols* 11, 2054–2065.
- IMOD/Etomo user guide: https://bio3d.colorado.edu/imod/doc/etomoTutorial.html
- teamtomo EMPIAR-10164 walkthrough: https://teamtomo.org/teamtomo-site-archive/walkthroughs/EMPIAR-10164/
- Warp documentation: https://www.warpem.com/warp/
