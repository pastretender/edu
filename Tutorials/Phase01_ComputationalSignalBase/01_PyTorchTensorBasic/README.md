# 01 — PyTorch Tensor Basics & Image I/O

## What you will learn

- What a tensor is and how it generalises scalars, vectors, and matrices
- Creating tensors with `torch.zeros`, `torch.ones`, `torch.rand`, and from NumPy arrays
- Moving tensors to a GPU (`tensor.to("cuda")`) and why that matters for training speed
- Setting random seeds for reproducibility
- Loading **DICOM** and **TIFF** scientific images, reading their metadata, normalising pixel values, and visualising with Matplotlib
- Basic volumetric (3-D) data operations

## Notebooks

| File | Description |
|------|-------------|
| `introduction_to_pytorch.ipynb` | PyTorch tensors from scratch: creation, arithmetic, indexing, broadcasting, and autograd overview |
| `basic_operations.ipynb` | Practical image I/O: DICOM metadata, TIFF normalisation, volume slicing |

## Key concepts

**Tensor** — the fundamental data structure in PyTorch. Think of it as a NumPy `ndarray` that can live on a GPU and tracks gradients automatically.

**DICOM** — the standard file format for clinical medical images (CT, MRI, X-ray). Each file embeds metadata (patient info, voxel spacing, acquisition parameters) alongside the pixel array.

**TIFF** — a lossless format widely used in microscopy. May be 16-bit or 32-bit float, so pixel values must be normalised before display.

## Setup

```bash
pip install torch torchvision imageio pydicom matplotlib numpy
```

All notebooks also include install cells you can run in Colab.

## Tips for beginners

- If `torch.cuda.is_available()` returns `False`, the notebooks still work on CPU — they will just be slower.
- Pay close attention to tensor **shape** (`tensor.shape`) and **dtype** (`tensor.dtype`) — mismatches cause the majority of beginner errors.
