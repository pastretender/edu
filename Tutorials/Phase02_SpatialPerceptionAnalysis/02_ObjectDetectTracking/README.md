# 02 — Object Detection & Tracking

## What you will learn

- How modern anchor-based detectors work: **YOLOv5** architecture (backbone, neck, head), anchor boxes, multi-scale detection
- Preparing a custom dataset in Roboflow's YOLOv5-PyTorch format and writing the YAML config file
- Key training hyperparameters: image size, batch size, epochs, depth/width multipliers
- Evaluating a detector: precision, recall, mAP
- **Centroid-based 3-D object tracking** with the **Linear Assignment Problem (LAP)** algorithm
- Using **LapTrack** and **napari** to track labelled structures in volumetric time-lapse microscopy

## Notebooks

| File | Description |
|------|-------------|
| `object_detection.ipynb` | Train YOLOv5 on a custom dataset end-to-end |
| `LAP_LapTrack.ipynb` | Compute 3-D label centroids and link them across time with LapTrack |

## Key concepts

**YOLO (You Only Look Once)** — a single-stage detector that divides the image into a grid and predicts bounding boxes and class scores simultaneously, making it fast enough for real-time use.

**Anchor boxes** — predefined bounding-box shapes at multiple scales that the detector refines. Choosing anchors that match the aspect-ratios of your objects improves convergence.

**mAP (mean Average Precision)** — the standard object-detection metric. Averages the area under the precision-recall curve across all classes and IoU thresholds.

**LAP (Linear Assignment Problem)** — optimal bipartite matching between detected objects in consecutive frames, minimising total displacement cost. Handles object appearance and disappearance.

**LapTrack** — a Python library that wraps LAP solving for multi-frame tracking; works on 2-D and 3-D data.

## Why tracking matters in biology

Live-cell imaging generates data where individual cells must be followed through time to measure division, migration, or signalling dynamics. Manual tracking is impractical at scale; automated centroid or feature-based linking is the standard approach.

## Setup

```bash
# YOLOv5 (clones its own repo with requirements)
git clone https://github.com/ultralytics/yolov5 && pip install -r yolov5/requirements.txt

# LapTrack
pip install laptrack napari
```
