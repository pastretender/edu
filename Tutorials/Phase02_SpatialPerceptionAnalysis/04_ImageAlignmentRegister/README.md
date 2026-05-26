# 04 — Image Alignment & Registration

## What you will learn

- What image registration is and why it is hard: the difference between rigid and deformable transforms
- Formulating registration as an **optimisation problem** over a similarity metric
- **SimpleITK** registration: specifying a transform, a metric, and an optimizer; running gradient descent; evaluating the result
- **Image-to-Image Translation** with **Pix2Pix** (conditional GAN): translating out-of-focus microscopy images into in-focus ones
  - U-Net generator architecture
  - PatchGAN discriminator and why it focuses on local texture rather than global structure
  - Paired vs. unpaired training

## Notebooks

| File | Description |
|------|-------------|
| `Image_Registration.ipynb` | Classical rigid registration with SimpleITK; mutual information metric; evaluation on a synthetic shift |
| `image_translation.ipynb` | Pix2Pix GAN pipeline for focus correction on real TIFF microscopy data |

## Key concepts

**Image registration** — finding a transformation T such that `T(moving_image) ≈ fixed_image`. Used whenever images were acquired at different times, angles, or modalities and need to be compared pixel-by-pixel.

**Similarity metric** — a scalar that measures how well two images are aligned. Common choices:
- Mean Squared Error (same modality, same contrast)
- Mattes Mutual Information (different modalities, e.g. MRI T1 vs T2)

**Pix2Pix** — a conditional GAN framework that learns a mapping from one image domain (out-of-focus) to another (in-focus) given paired training examples. The generator is a U-Net; the discriminator evaluates 70×70 patches rather than the whole image.

**TIFF image loading** — microscopy data is often stored as 16-bit TIFF. The `image_translation.ipynb` notebook shows how to normalise these to the [0,1] range expected by neural networks.

## Practical context

Registration is a prerequisite for longitudinal studies (tracking a tumour across CT scans), multi-stain analysis, and cryo-ET subtomogram alignment. Image-to-image translation is increasingly used to synthesise missing imaging modalities or correct acquisition artefacts.

## Setup

```bash
pip install SimpleITK torch torchvision tifffile matplotlib numpy
```
