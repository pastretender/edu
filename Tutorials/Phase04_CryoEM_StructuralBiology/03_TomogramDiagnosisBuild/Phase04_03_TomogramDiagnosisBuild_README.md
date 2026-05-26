# 03 — Build & Diagnose a Cryo-ET Tomogram

## Overview

This is a **hands-on practical** rather than a Jupyter notebook. You will process a real
cryo-ET tilt series from the public dataset **EMPIAR-10164** using IMOD/Etomo — the same
software stack used by structural-biology labs worldwide. You will build two tomographic
reconstructions, systematically diagnose their quality, and answer a set of guided
questions that connect the computational theory from Modules 01 and 02 to real experimental
data.

This practical bridges the gap between "I understand the algorithm in a notebook" and "I
can operate production software on real data."

---

## Prerequisites

**Complete Modules 01 and 02 first.** The concepts you will encounter here — CTF, missing
wedge, WBP, SIRT, fiducial residuals — are all introduced computationally in those modules.
Arriving here without that background will make the IMOD dialogue boxes difficult to
interpret.

**Software to install before starting:**
- **IMOD** (includes Etomo and 3dmod): https://bio3d.colorado.edu/imod/download.html
  - Linux (Ubuntu 20.04+): install via the provided shell script
  - macOS: installer DMG available; requires Rosetta 2 on Apple Silicon
  - Windows: not natively supported — use **WSL2** with Ubuntu, or a Linux VM
- **Warp** (optional but recommended): https://www.warpem.com
  - Handles motion correction of multi-frame tilt movies before IMOD stacking
  - If unavailable, the IMOD-only path is described in the practical document

---

## Files

| File | Description |
|------|-------------|
| `build_and_diagnose_tomogram.md` | Step-by-step instructions for all three steps (stack → align → reconstruct), four diagnostic questions with answer templates, and guidance on what good vs bad results look like |

---

## Dataset: EMPIAR-10164

**EMPIAR** (Electron Microscopy Public Image Archive) is the community repository for raw
cryoEM and cryo-ET data, analogous to the PDB for atomic coordinates.

EMPIAR-10164 contains cryo-ET data of **Mycoplasma pneumoniae** — a small bacterial cell
whose cytoplasmic membrane, ribosomes, and surface protein complexes are visible in the
reconstructed tomograms.

For this practical you will work with **TS_01** (tilt series 01), one of a 5-tilt-series
subset that the teamtomo walkthrough provides a download script for:

```
https://teamtomo.org/teamtomo-site-archive/walkthroughs/EMPIAR-10164/preparation.html
```

The download consists of:
- Per-tilt multi-frame `.mrc` files (raw movies, one per tilt angle)
- SerialEM `.mdoc` metadata files (acquisition parameters, tilt angles, defocus estimates)

Total download size for TS_01: ~8–12 GB.

---

## The three processing steps

### Step A: Build the tilt stack

CryoET data is acquired as a **tilt series** — the specimen is imaged at many angles
(typically −60° to +60° in 3° increments = ~40 images) to provide the angular coverage
needed for 3-D reconstruction.

Each tilt is recorded as a **movie** of multiple frames (the detector's movie mode) that
must first be aligned and averaged to correct beam-induced sample motion. This step
produces a single corrected image per tilt angle. The corrected images are then stacked
into a single `.st` file that Etomo can process.

**With Warp:** Follow the teamtomo walkthrough "Stack generation" section. Warp performs
motion correction and exports an IMOD-compatible stack directly.

**IMOD-only path:** Use `alignframes` for per-frame alignment and `newstack` to assemble
the final stack. Quality will be slightly lower (Warp's motion correction is more
sophisticated) but the reconstruction pipeline is identical.

### Step B: Align the tilt series (fiducial tracking)

The tilt stack is a sequence of 2-D images, but each image was recorded at a different
stage position. Before reconstruction, all images must be aligned to a common coordinate
system. EMPIAR-10164 uses **colloidal gold fiducials** — tiny gold spheres attached to
the specimen that appear as high-contrast dots in the images and can be tracked across all
tilt angles.

In Etomo:
1. Choose **Fiducial Model Based** alignment
2. Pick fiducials manually (or automatically) in the first tilt
3. Use the automated **Fiducial Tracking** to follow beads through the tilt range
4. Iterate the tracking until the **mean residual** (average deviation of tracked bead
   positions from the fitted model) stabilises

**Target residual:** < 1 pixel is acceptable; < 0.5 pixels is good. High-tilt images
(> ±50°) typically have worse residuals because the foreshortened geometry stretches the
fiducials and makes them harder to track accurately.

The alignment produces a transformation file that maps each tilt image into the common
coordinate frame. This is used during reconstruction.

### Step C: Reconstruct two tomograms

Reconstruct the aligned tilt series using two complementary algorithms:

**Weighted Back Projection (WBP):**
- Direct application of the Fourier Slice Theorem (see Module 02)
- Each CTF-corrected tilt is back-projected into 3-D space and summed
- Fast and linear; the gold standard for high-resolution work
- Produces streaking artefacts from high-contrast objects (gold beads, carbon edges)
- The `radial` weighting filter in Etomo applies the standard WBP ramp in Fourier space

**Simultaneous Iterative Reconstruction Technique (SIRT):**
- Iterative refinement: start from a zero volume, forward-project, compare to data,
  back-project the difference, update the volume; repeat ~10–50 iterations
- Substantially reduces streaking from high-contrast features
- Produces smoother, lower-contrast results — better for visual inspection and segmentation
- Much slower than WBP; less suitable for high-resolution sub-tomogram averaging

Reconstruct both, save them as separate `.rec` files, and compare them in 3dmod.

---

## Diagnostic questions

You must answer all four questions in `build_and_diagnose_tomogram.md`.

### Q1 — Alignment quality

After fiducial tracking converges in Etomo, record:
- The **final mean residual** in pixels
- Which tilt angles contribute the **largest per-tilt residuals** and why (beam-induced
  specimen drift at high tilt? Poor bead tracking at the tilt extremes?)
- Whether high-tilt or low-tilt images are more problematic, and the physical explanation

Expected: mean residual < 0.5 px is good; > 1.5 px suggests a tracking problem.

### Q2 — Missing-wedge artefact

In 3dmod's Slicer tool, rotate the reconstructed tomogram around the tilt axis and observe
direction-dependent elongation of spherical objects (gold beads, ribosome densities).

Then reconstruct a second time after **excluding all tilts beyond ±40°** (Etomo's
"Exclude Views" option) to exaggerate the missing wedge. Compare:
- Which structural features distort most as angular coverage worsens
- The artifact in the Z (beam) direction vs the XY plane
- Whether membranes, ribosomes, or gold beads are most affected, and why

This connects directly to the Fourier-space missing-wedge visualisation in Module 02.

### Q3 — WBP vs SIRT

Pick the same region of interest (a membrane patch, a cluster of ribosomes, and a gold
bead) in both reconstructions and compare:
- **Streaking** — which method shows less linear artefacting near high-contrast objects?
- **Sharpness** — which looks crisper? Which looks smoother?
- **Contrast** — how do membrane densities compare between the two methods?
- **Practical choice** — which would you trust for template matching (finding ribosomes
  systematically)? Which for manual or automated segmentation?

The standard answer: WBP for sub-tomogram averaging (template matching), SIRT for
visualisation and segmentation — but the notebook asks you to justify this from the data.

### Q4 — Pixel size

In Etomo, note the **pixel size** reported for the final reconstructed tomogram. Explain:
- How this relates to the physical pixel size on the detector (in microns)
- How the magnification and binning factor affect it
- What binning 2× or 4× costs you in resolution and gains you in computation and contrast

---

## How to view the tomogram in 3dmod

```bash
# Open the reconstruction
3dmod TS_01_WBP.rec

# In 3dmod:
#   Ctrl+T  → open Slicer (for arbitrary-angle cross-sections)
#   Ctrl+G  → open 3dmod model window (for segmentation)
#   Middle-click drag in Slicer → rotate the slab
#   Scroll wheel → move through Z slices in the main window
```

The Slicer is essential for Q2. Set the slab thickness to ~5 nm, orient it perpendicular
to the tilt axis, and watch features elongate as you rotate toward the beam direction.

---

## Troubleshooting

| Problem | Likely cause | Solution |
|---------|-------------|---------|
| Fiducial tracking fails repeatedly | Wrong bead radius set | Open a tilt image in 3dmod, measure the bead diameter in pixels, re-enter in Etomo |
| Residuals do not converge below 2 px | Stage drift between tilts; poor bead distribution | Exclude extreme tilts; add more manually tracked beads; check `.mdoc` for large stage-shift jumps |
| Reconstruction is all black | Wrong input file path or wrong orientation | Verify the aligned stack `.ali` file is non-empty; check Etomo's axis orientation setting |
| 3dmod crashes on opening the `.rec` | File is incomplete | Check that the reconstruction job finished; re-run if the log shows an error |
| IMOD not found after installation | PATH not updated | Add `source /etc/imod-setup.sh` (or the equivalent for your platform) to `.bashrc` |

---

## Further reading

- IMOD/Etomo user guide: https://bio3d.colorado.edu/imod/doc/etomoTutorial.html
- teamtomo EMPIAR-10164 walkthrough: https://teamtomo.org/teamtomo-site-archive/walkthroughs/EMPIAR-10164/
- Warp documentation: https://www.warpem.com/warp/
- Mastronarde, D.N. (1997). "Dual-axis tomography: an approach with alignment methods that
  preserve resolution." *Journal of Structural Biology* 120, 343–352. — The original paper
  introducing the fiducial alignment approach implemented in IMOD.
