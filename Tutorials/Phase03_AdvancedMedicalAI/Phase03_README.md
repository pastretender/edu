# Phase 03 — Advanced Medical AI

**Advanced level.** You should be comfortable with deep-learning training loops, image
segmentation, and registration concepts from Phase 02 before starting here. If you have not
yet worked through Phase 01 (PyTorch, Fourier filtering) and Phase 02 (segmentation,
classical registration), go there first.

This phase brings together the full toolkit of modern clinical and research medical-imaging
AI. Each module is a self-contained deep dive, but they are ordered so that later modules can
build on intuitions developed in earlier ones.

---

## Contents

| Subfolder | Topic | Notebooks |
|-----------|-------|-----------|
| [`01_Volume3DSegmentation`](01_Volume3DSegmentation/) | 3-D U-Net spleen segmentation with MONAI on the Medical Segmentation Decathlon | `3D_Medical_Image_Segmentation_MONAI.ipynb` |
| [`02_MultimodalRegistration`](02_MultimodalRegistration/) | Classical (SimpleITK, ANTs) and deep-learning (VoxelMorph-style) multimodal image registration | `multimodal_image_registration.ipynb` |
| [`03_MedicalGenerativeModel`](03_MedicalGenerativeModel/) | Cross-modal MRI synthesis using Schrödinger Bridge Conditional Flow Matching | `SB_CFM_Medical_Synthesis.ipynb` |
| [`04_MedicalVisionLanguage`](04_MedicalVisionLanguage/) | BiomedCLIP zero-shot classification, BLIP-2 captioning, and LLaVA-1.5 visual question answering on medical images | `multimodal_medical_imaging_tutorial.ipynb` |

---

## Why this phase matters

Phases 01 and 02 covered the foundations: tensors, filters, features, segmentation, and
registration. Phase 03 is where those building blocks converge into the kinds of systems
being deployed in research labs and (increasingly) in clinical pipelines today.

The four modules are not independent curiosities — they address the same overarching
challenge from four angles:

- **Module 01** — given a 3-D scan, *delineate* the structure of interest automatically.
- **Module 02** — given scans from different instruments or time points, *align* them into
  a common space so they can be compared.
- **Module 03** — given only one imaging modality, *synthesise* a missing one rather than
  requiring the patient to return for an additional scan.
- **Module 04** — given a medical image and a natural-language query, *answer* it — zero-shot,
  without task-specific training data.

Together these modules cover the three core paradigms of modern medical vision AI:
discriminative models (segmentation), transformation models (registration), and generative
models (synthesis and vision-language).

---

## Learning goals

After finishing this phase you will be able to:

- Set up MONAI's patch-based 3-D training pipeline and train a volumetric U-Net to
  convergence on a public CT benchmark
- Explain why Mutual Information is the standard similarity metric for multimodal
  registration, and implement it as a differentiable loss
- Apply classical rigid (SimpleITK) and deformable (ANTs/SyN) registration, and compare
  the results to a trained deep-learning registration network
- Derive the Conditional Flow Matching objective, understand what the Schrödinger Bridge
  modification adds, and train a synthesis model on paired 2-D MRI slices
- Run BiomedCLIP zero-shot classification, BLIP-2 image captioning, and LLaVA VQA on
  medical images; evaluate generated text with BERTScore, BLEU, and ROUGE; and manage
  multiple large models within a 16 GB VRAM budget

---

## Prerequisites

- **Phases 01 and 02 completed**, or equivalent experience with PyTorch, training loops,
  DataLoaders, and convolutional segmentation architectures
- Comfort with reading and modifying Python class definitions
- Basic familiarity with loss functions, gradient descent, and learning rate schedules

---

## Hardware requirements

All notebooks are designed for **Google Colab with a T4 GPU (16 GB VRAM)**.

| Module | VRAM usage | Notes |
|--------|-----------|-------|
| 01 — 3-D Segmentation | ~10 GB | Mixed-precision brings this within T4 limits |
| 02 — Registration | ~4 GB | Deep-learning registration section only |
| 03 — Generative Model | ~6 GB | 2-D slices keep memory manageable |
| 04 — Vision-Language | ~14 GB peak | Models must be loaded and unloaded one at a time |

For Module 04, follow the notebook's explicit teardown pattern between model loads
(`del model`, `gc.collect()`, `torch.cuda.empty_cache()`). Skipping teardown will cause
out-of-memory errors when loading the next model.

---

## How the modules connect

```
Phase 02 segmentation ──► Module 01 (3-D U-Net)
Phase 02 registration ──► Module 02 (Multimodal registration)
                               │
                               ▼
                         Module 03 (Synthesis: generates a missing modality
                                    so registration can run monomodally)
                               │
                               ▼
                         Module 04 (VLMs: can answer questions about
                                    the segmented / registered / synthesised outputs)
```

---

## Suggested order

`01_Volume3DSegmentation` → `02_MultimodalRegistration` → `03_MedicalGenerativeModel` → `04_MedicalVisionLanguage`

Modules 03 and 04 are largely independent of each other and can be swapped once you have
finished 01 and 02.

---

## Further reading

- MONAI documentation: https://docs.monai.io
- VoxelMorph paper (Balakrishnan et al., 2019): https://arxiv.org/abs/1809.05231
- Flow Matching (Lipman et al., 2022): https://arxiv.org/abs/2210.02747
- BiomedCLIP (Zhang et al., 2023): https://arxiv.org/abs/2303.00915
- BLIP-2 (Li et al., 2023): https://arxiv.org/abs/2301.12597
