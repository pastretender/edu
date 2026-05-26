# Phase 03 — Advanced Medical AI

**Level: Advanced** · Complete Phases 01 and 02 (or have equivalent experience with PyTorch training loops, U-Net segmentation, and image registration) before starting here.

Phase 03 is where the building blocks become systems. Each module targets a different axis of the same overarching problem — turning multi-modal, multi-session clinical imaging data into actionable information — and the four modules are deliberately ordered so that later ones can use the outputs of earlier ones.

```
Delineate (Module 01 — 3-D segmentation)
     ↓
Align (Module 02 — multimodal registration)
     ↓
Synthesise (Module 03 — generate a missing modality)
     ↓
Interpret (Module 04 — ask a question in natural language)
```

---

## Contents

| # | Subfolder | Core topic | Notebook | GPU needed? |
|---|-----------|-----------|---------|-------------|
| 1 | [`01_Volume3DSegmentation`](01_Volume3DSegmentation/) | 3-D U-Net spleen segmentation with MONAI on MSD | `3D_Medical_Image_Segmentation_MONAI.ipynb` | Yes |
| 2 | [`02_MultimodalRegistration`](02_MultimodalRegistration/) | Classical (SimpleITK, ANTs) + deep (VoxelMorph-style) registration | `multimodal_image_registration.ipynb` | Yes (deep section) |
| 3 | [`03_MedicalGenerativeModel`](03_MedicalGenerativeModel/) | T1→T2 MRI synthesis via Schrödinger Bridge CFM | `SB_CFM_Medical_Synthesis.ipynb` | Yes |
| 4 | [`04_MedicalVisionLanguage`](04_MedicalVisionLanguage/) | BiomedCLIP, BLIP-2, LLaVA-1.5 on real medical images + Gradio demo | `multimodal_medical_imaging_tutorial.ipynb` | Yes |

---

## Module Summaries

### Module 01 — 3-D Medical Image Segmentation with MONAI

The direct volumetric extension of Phase 02's 2-D U-Net. The target task is spleen segmentation on the Medical Segmentation Decathlon (Task09_Spleen) — 41 abdominal CT scans with binary labels, a publicly available benchmark with comparable literature results.

The technical challenge that distinguishes 3-D from 2-D segmentation: a full abdominal CT at 1 mm isotropic resolution is approximately 200 × 512 × 512 voxels (~100 M floats) and does not fit in GPU memory. The module covers MONAI's solution in full: a patch-based training strategy with `RandCropByPosNegLabeld` that ensures equal sampling from spleen and background patches, `CacheDataset` to eliminate the NIfTI decompression bottleneck in subsequent epochs, mixed-precision training (AMP) to halve VRAM usage, and sliding-window inference to evaluate the full volume at test time.

The architecture is a 3-D U-Net with skip connections across all four encoder levels. Loss is `DiceCELoss` — a weighted combination of Dice (robust to the severe class imbalance: spleen is ~1% of all voxels) and cross-entropy (stable gradient signal early in training). Evaluation uses Dice Similarity Coefficient (DSC) and 95th-percentile Hausdorff Distance (HD95).

**Key skills:** NIfTI affine matrices, MONAI transform pipeline (`Orientationd`, `Spacingd`, `ScaleIntensityRanged`), `CacheDataset`, `DiceCELoss`, `AdamW` + cosine LR, `torch.cuda.amp.autocast`, `sliding_window_inference`, DSC, HD95.

**Target performance:** DSC > 0.90 after 50 epochs on a T4 GPU.

---

### Module 02 — Multimodal Image Registration

Registration finds the transform that maps one image into alignment with another. When the images come from different modalities (CT ↔ MRI, T1 ↔ T2, MRI ↔ PET) this is *multimodal registration*, and it demands metrics that capture anatomical correspondence without assuming any particular intensity relationship between modalities.

The module builds a three-level comparison using a synthetic brain phantom with known ground-truth misalignment:

**Metric comparison** — MSE, NCC, and Mutual Information are evaluated as a function of misalignment angle. This makes MI's superiority for multimodal tasks immediately visible: MSE reports maximum dissimilarity for a perfectly aligned T1/T2 pair; MI correctly identifies alignment.

**Classical registration (SimpleITK)** — rigid → affine → B-spline deformable registration, with a mandatory multi-resolution pyramid at each stage. A common mistake (starting deformable registration from scratch rather than warm-starting from an affine transform) is demonstrated and its consequences shown.

**SyN deformable registration (ANTs)** — the symmetric diffeomorphic normalisation algorithm widely used in neuroimaging, which produces a physically valid, invertible displacement field with positive Jacobian determinant everywhere.

**Deep registration (VoxelMorph-style)** — a U-Net that takes the concatenated fixed and moving images as input and outputs a dense displacement field. A spatial transformer (differentiable `grid_sample`) applies the field during training so the alignment loss back-propagates through the warp. The smoothness regularisation weight is shown to be the key hyperparameter: too small produces chaotic displacement fields; too large under-registers.

Evaluation covers propagated-segmentation Dice, Target Registration Error on anatomical landmarks, and Jacobian determinant histograms for deformable methods.

**Key skills:** Mattes MI, NMI, `sitk.ImageRegistrationMethod`, multi-resolution pyramid, ANTs SyN, `torch.nn.functional.grid_sample`, spatial transformer, Jacobian determinant.

---

### Module 03 — Medical Generative Models: Cross-Modal Synthesis

A patient who received a CT scan did not necessarily receive an MRI. A rare pathology may appear on only one of the available modalities. Cross-modal synthesis generates a plausible image in a target modality from a source, effectively filling in the missing scan without additional acquisition.

The module implements **Schrödinger Bridge Conditional Flow Matching (SB-CFM)** — a state-of-the-art generative framework that is substantially more stable and sample-efficient than GANs, and more computationally direct than standard diffusion models.

**Conditional Flow Matching (CFM)** trains a neural network to predict a time-dependent vector field that transports samples from a source distribution (T1) to a target distribution (T2). The training loss is a simple regression: `‖v_θ(x_t, t) – (x₁ – x₀)‖²` — no discriminator, no score matching. At inference, integrate the ODE from t=0 to t=1 starting from the source image.

**The Schrödinger Bridge** modification selects the minimum-entropy transport path between the two distributions, producing straighter ODE trajectories that require fewer integration steps at inference and have lower-variance training targets.

The architecture is MONAI's `BasicUNet` with a sinusoidal time embedding at the bottleneck and the source image concatenated channel-wise with the interpolated input. Two inference solvers are compared (Euler and RK2), and the module ends with a rare-pathology augmentation demo — the clinical use case that motivates the whole approach.

The dataset is IXI paired T1/T2 brain MRI, which is mathematically equivalent to MRI→CT transport and freely available without institutional access.

**Key skills:** `torchcfm.SchrodingerBridgeConditionalFlowMatcher`, sinusoidal time embeddings, Euler / RK2 ODE integration, PSNR, SSIM, MAE, `einops.rearrange`, rare-pathology oversampling.

---

### Module 04 — Medical Vision-Language Models

The previous three modules required training or fine-tuning a model on medical data. This module asks how much pre-trained foundation models can do *without* any additional training — and where they fall short.

Three open-source multimodal models are run on four real medical image types (chest X-ray, brain MRI, retinal fundus, histopathology):

**BiomedCLIP** — a CLIP model pre-trained on 15 million biomedical image-text pairs from PubMed Central. Used for zero-shot classification: embed the image with the ViT encoder, embed candidate class labels with the text encoder, and take the argmax cosine similarity. Prompt sensitivity is demonstrated explicitly: the same image can yield dramatically different confidence scores with different prompt wording.

**BLIP-2** — a frozen ViT-L (307 M params) coupled to a frozen FlanT5-XL (3 B params) via a trainable Querying Transformer (~188 M). Only the Q-Former was trained, making BLIP-2 highly parameter-efficient. Applied to captioning all four image types; the module shows where the model produces plausible clinical language and where it hallucinates anatomically confident but factually wrong descriptions.

**LLaVA-1.5-7B** — a CLIP ViT connected to Vicuna-7B via a two-layer MLP, loaded in 4-bit quantisation (~5 GB VRAM) to fit alongside the other models on a T4. Used for open-ended visual question answering with clinical prompts.

Generated text is evaluated with BERTScore (semantic similarity using BERT token embeddings), SacreBLEU (n-gram precision), and ROUGE-L (longest common subsequence recall). The module closes with a Gradio demo that wraps all three models in a single web interface.

**Key skills:** `open_clip_torch`, zero-shot classification prompt design, BLIP-2 Q-Former architecture, 4-bit quantisation with `bitsandbytes`, `BERTScore`, `sacrebleu`, CUDA memory management (`del model`, `gc.collect()`, `torch.cuda.empty_cache()`), `gradio`.

**Critical caveat:** None of these models carry regulatory clearance. They are research tools and will hallucinate. Always validate against ground-truth reports.

---

## Learning Goals

After completing Phase 03 you will be able to:

- Set up MONAI's patch-based 3-D training pipeline and train a volumetric U-Net on a public CT benchmark to clinically useful Dice (> 0.90)
- Explain why Mutual Information is the correct metric for multimodal registration and implement it as a differentiable loss
- Apply rigid, affine, and deformable registration using SimpleITK and ANTs, and compare to a trained deep-learning registration network on the same dataset
- Derive the Conditional Flow Matching objective, understand what the Schrödinger Bridge modification adds in terms of trajectory efficiency, and train a synthesis model on paired 2-D MRI slices
- Run BiomedCLIP zero-shot classification, BLIP-2 captioning, and LLaVA VQA on medical images; evaluate generated text quantitatively; and manage multiple large models within a 16 GB VRAM budget

---

## Prerequisites

- **Phases 01 and 02 completed** — or equivalent experience with PyTorch, training loops, DataLoaders, U-Net segmentation, and image-registration concepts
- Comfort with reading and modifying Python class definitions
- Familiarity with loss functions, gradient descent, and learning rate schedulers

---

## Hardware Requirements

All notebooks target **Google Colab with a T4 GPU (16 GB VRAM)**.

| Module | Peak VRAM | Notes |
|--------|----------|-------|
| 01 — 3-D Segmentation | ~10 GB | Mixed-precision brings this within T4 limits |
| 02 — Registration | ~4 GB | Deep-learning section only; classical sections are CPU |
| 03 — Generative Model | ~6 GB | 2-D slices keep memory manageable |
| 04 — Vision-Language | ~14 GB peak | **Load and unload one model at a time** — see teardown pattern below |

**Module 04 teardown pattern (mandatory):**

```python
del model, processor      # Remove Python references
import gc; gc.collect()   # Python garbage collector
torch.cuda.empty_cache()  # Release CUDA memory
```

Skipping any of these three steps will cause OOM when loading the next model.

---

## Setup

```bash
# Each notebook also installs its own dependencies in the first cell
pip install "monai[nibabel,tqdm]" SimpleITK antspyx torchcfm nibabel \
    "transformers>=4.37.0" "accelerate>=0.27.0" "bitsandbytes>=0.41.0" \
    open_clip_torch bert-score sacrebleu "gradio>=4.20.0" \
    torch torchvision einops matplotlib scikit-image "numpy<2.0.0"
```

---

## Suggested Order

```
01_Volume3DSegmentation
         ↓
02_MultimodalRegistration
         ↓
03_MedicalGenerativeModel    ←→    04_MedicalVisionLanguage
```

Modules 03 and 04 are largely independent of each other. Once you have finished 01 and 02, they can be taken in either order.

---

## How the Modules Connect in Practice

The four modules address the same clinical challenge from different angles:

1. A 3-D scan arrives → **Module 01** delineates the structure of interest automatically
2. Serial scans from different sessions arrive → **Module 02** aligns them into a common space
3. A scan in the needed modality is missing → **Module 03** synthesises it from an available modality, enabling monomodal registration with MSE (simpler and faster than multimodal MI)
4. A clinician wants to query the processed data in natural language → **Module 04** answers zero-shot, without task-specific fine-tuning

Together they cover the three core paradigms of modern medical vision AI: discriminative models (segmentation), transformation models (registration + synthesis), and generative + language models (VLMs).

---

## Further Reading

- MONAI documentation: https://docs.monai.io
- VoxelMorph (Balakrishnan et al., 2019): https://arxiv.org/abs/1809.05231
- Flow Matching (Lipman et al., 2022): https://arxiv.org/abs/2210.02747
- Schrödinger Bridge CFM (Liu et al., 2023): https://arxiv.org/abs/2303.00585
- BiomedCLIP (Zhang et al., 2023): https://arxiv.org/abs/2303.00915
- BLIP-2 (Li et al., 2023): https://arxiv.org/abs/2301.12597
- LLaVA-1.5 (Liu et al., 2023): https://arxiv.org/abs/2310.03744
