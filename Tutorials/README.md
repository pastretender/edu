# Tutorials — Learning Guide

This directory is a four-phase journey from foundational computational skills to advanced medical and structural-biology AI. Each phase builds on the one before it; a complete beginner should work through them in order.

---

## 🗺️ Roadmap

| Phase | Folder | Theme | Good for |
|-------|--------|-------|----------|
| 1 | [`Phase01_ComputationalSignalBase`](Phase01_ComputationalSignalBase/) | PyTorch basics, image filtering & classical feature extraction | Absolute beginners |
| 2 | [`Phase02_SpatialPerceptionAnalysis`](Phase02_SpatialPerceptionAnalysis/) | Dimensionality reduction, detection, segmentation & registration | Intermediate learners |
| 3 | [`Phase03_AdvancedMedicalAI`](Phase03_AdvancedMedicalAI/) | 3-D segmentation, multimodal registration, generative models & vision-language models | Advanced learners with a medical-imaging interest |
| 4 | [`Phase04_CryoEM_StructuralBiology`](Phase04_CryoEM_StructuralBiology/) | Low-SNR processing, 3-D reconstruction & real tomogram workflows for cryo-EM | Structural biology / cryo-EM audience |

---

## Prerequisites

- **Python 3.9+** and familiarity with NumPy / Matplotlib
- **PyTorch** — Phase 1 covers the basics from scratch
- **Google Colab** (free GPU) works for all notebooks; Phase 3 & 4 require a GPU

---

## How to use these notebooks

1. Start with Phase 1 even if you are comfortable with Python — the PyTorch tensor basics lay the vocabulary used throughout.
2. Each subfolder has its own `README.md` explaining the notebooks inside, what you will learn, and any setup steps.
3. Run cells top-to-bottom; most notebooks install their own dependencies in the first cell.
4. The phases are **roughly** independent after Phase 1, so you can jump to the area that interests you most.

---

## Quick-start

```bash
# Clone and open locally
git clone https://github.com/pastretender/edu.git
cd edu/Tutorials
jupyter lab
```

Or open any notebook directly in Google Colab by replacing `github.com` with `colab.research.google.com/github` in the URL.
