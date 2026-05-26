# 01 — PyTorch Tensor Basics & Image I/O

## Overview

This module is the entry point for the entire tutorial series. Before you can train a
segmentation model, simulate a cryoEM micrograph, or register two MRI scans, you need a
reliable way to represent, move, and manipulate multidimensional numerical data on a GPU.
That is what this module is about.

By the end you will be fluent with PyTorch tensors — the data structure that underlies
every later module — and you will know how to load the two scientific image formats
(DICOM and TIFF) that appear throughout Phases 01–04.

---

## What you will learn

### Tensors: the fundamental data structure

A **tensor** is an n-dimensional array that lives either on CPU or GPU memory and
optionally tracks gradients for automatic differentiation. It generalises the familiar
hierarchy:

```
Scalar  (0-D tensor)  →  a single number, e.g. tensor(3.14)
Vector  (1-D tensor)  →  a sequence,     e.g. tensor([1, 2, 3])
Matrix  (2-D tensor)  →  a table,        e.g. tensor([[1, 2], [3, 4]])
Volume  (3-D tensor)  →  a 3-D array,    e.g. a CT scan of shape (D, H, W)
```

In medical imaging, images are almost always represented as 4-D or 5-D tensors:
`(Batch, Channel, Height, Width)` or `(Batch, Channel, Depth, Height, Width)`. Understanding
this layout prevents the majority of shape-mismatch errors you will encounter in later phases.

**Creation patterns you must know:**

| Call | What it produces |
|------|-----------------|
| `torch.zeros(3, 4)` | All-zeros matrix of shape (3, 4) |
| `torch.ones(2, 3, 4)` | All-ones 3-D tensor |
| `torch.rand(8, 8)` | Uniform random values in [0, 1) |
| `torch.randn(8, 8)` | Standard normal (mean 0, std 1) |
| `torch.from_numpy(arr)` | Zero-copy view of a NumPy array |
| `torch.arange(0, 10, 2)` | [0, 2, 4, 6, 8] |

**The two most important tensor attributes:**

- `tensor.shape` — the size along each dimension, e.g. `torch.Size([2, 512, 512])`. Print
  this at every stage of your pipeline. Most bugs are shape bugs.
- `tensor.dtype` — the numeric type, e.g. `torch.float32`, `torch.uint8`, `torch.int64`.
  Mismatched dtypes cause silent precision loss or hard errors in arithmetic operations.

### Indexing and broadcasting

**Indexing** works exactly like NumPy: `t[0]`, `t[:, 2:5]`, `t[..., 0]`, boolean masks.

**Broadcasting** is the rule that allows tensors of different shapes to be added or
multiplied as long as their shapes are compatible (dimensions align when trailing-padded
with 1s). For example, subtracting a per-channel mean vector of shape `(3,)` from an
image of shape `(3, 512, 512)` works without writing a loop. This is used constantly in
normalisation and loss computation.

### GPU acceleration

Modern deep-learning pipelines move tensors to a GPU (or MPS on Apple Silicon) to exploit
parallel hardware:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
t = t.to(device)           # move to GPU
t = t.cpu()                # move back to CPU
t = t.numpy()              # convert to NumPy (CPU only)
```

`torch.cuda.is_available()` returns `False` on Colab CPU runtimes — every notebook in
Phase 01 falls back to CPU automatically and still runs correctly, just more slowly.

### Reproducibility: random seeds

Any experiment that uses random numbers must fix a seed before it starts, or results will
differ between runs:

```python
torch.manual_seed(42)
np.random.seed(42)
```

In GPU experiments, add `torch.backends.cudnn.deterministic = True` and
`torch.backends.cudnn.benchmark = False` for full reproducibility at some speed cost.

### Autograd basics

PyTorch records all tensor operations when `requires_grad=True` is set, building a
computation graph. Calling `.backward()` on a scalar loss propagates gradients
automatically. You will not hand-write any gradients in this phase, but understanding that
`loss.backward()` fills `param.grad` for every parameter in the graph is essential before
Phase 02 introduces full training loops.

### DICOM: the clinical image format

**DICOM** (Digital Imaging and Communications in Medicine) is the international standard
for clinical medical images: CT, MRI, PET, X-ray. Every pixel array is paired with a
header containing hundreds of metadata fields, the most important being:

| Tag | Meaning | Why it matters |
|-----|---------|---------------|
| `PixelSpacing` | Physical size of each pixel (mm) | Required for distance and volume calculations |
| `SliceThickness` | Physical depth of each slice (mm) | Needed to construct an isotropic 3-D volume |
| `RescaleSlope`, `RescaleIntercept` | Hounsfield unit conversion | Must apply before using pixel values as densities |
| `Rows`, `Columns` | Image dimensions | Sanity-check after loading |

The `pydicom` library loads a `.dcm` file and returns a dataset where you access tags as
attributes. The pixel array is in `ds.pixel_array`; apply the Hounsfield transform with
`ds.RescaleSlope * ds.pixel_array + ds.RescaleIntercept`.

### TIFF: the microscopy format

**TIFF** (Tagged Image File Format) is ubiquitous in fluorescence and electron microscopy.
Unlike DICOM it carries minimal standardised metadata, but it supports:

- **16-bit and 32-bit float** pixel depth — values are not in [0, 255] and must be
  normalised before display or model input
- **Multi-page TIFFs** — a stack of images (a Z-series or time-series) stored in one file
- **Tiles and strips** — large images written in chunks for memory-efficient access

Use `tifffile` for reading scientific TIFFs (preserves bit depth and multi-page structure);
use `PIL.Image` only for display-ready 8-bit TIFFs.

Normalisation before display or model input:
```python
img = tifffile.imread("scan.tif").astype(np.float32)
img = (img - img.min()) / (img.max() - img.min() + 1e-8)
```

### Volumetric (3-D) operations

Medical images are 3-D: a CT scan is a stack of 2-D axial slices. In PyTorch a volume is
represented as a 5-D tensor `(B, C, D, H, W)`. The module covers:

- Slicing along any axis: `volume[0, 0, d, :, :]` gives the `d`-th axial slice
- Multi-planar reformatting: slicing along axis 2 (axial), axis 3 (coronal), axis 4 (sagittal)
- Basic morphological operations using `torch.nn.functional.max_pool3d` for downsampling

---

## Notebooks

| File | Description |
|------|-------------|
| `introduction_to_pytorch.ipynb` | PyTorch tensors from scratch: creation, arithmetic, indexing, broadcasting, GPU transfer, autograd overview, and reproducibility |
| `basic_operations.ipynb` | Practical image I/O: DICOM metadata inspection and Hounsfield conversion, TIFF 16-bit loading and normalisation, volume slicing and multi-planar visualisation |

---

## Setup

```bash
pip install torch torchvision imageio pydicom tifffile matplotlib numpy
```

All notebooks include install cells that run automatically on Google Colab.

---

## Common errors and how to fix them

| Error | Likely cause | Fix |
|-------|-------------|-----|
| `RuntimeError: expected scalar type Float but found Double` | NumPy default `float64` passed to a model expecting `float32` | Add `.float()` or `.to(torch.float32)` after `from_numpy()` |
| `RuntimeError: Expected all tensors to be on the same device` | One tensor is on CPU, another on GPU | Move all tensors to the same device before any operation |
| `Can't call numpy() on Tensor that requires grad` | Autograd is active | Detach first: `tensor.detach().cpu().numpy()` |
| DICOM pixel values look wrong | Forgot Hounsfield rescale | Apply `slope * pixel_array + intercept` |
| TIFF displayed as all-black | 16-bit values not normalised | Normalise to [0,1] before `plt.imshow()` |

---

## Tips for beginners

- Print `tensor.shape` and `tensor.dtype` compulsively. Every time you pass a tensor to a
  function, check these two attributes first. This single habit prevents the majority of
  beginner bugs.
- `tensor.squeeze()` removes dimensions of size 1; `tensor.unsqueeze(0)` adds one. Adding
  a batch dimension before passing to a model (`img.unsqueeze(0)`) is the most common
  single-line fix.
- DICOM files often store uint16 pixel arrays. Always cast to float32 before arithmetic:
  `ds.pixel_array.astype(np.float32)`.
- A Colab T4 session has ~15 GB VRAM. For Phase 01 you will never need more than a few MB
  — do not worry about GPU memory management until Phase 02 and beyond.
