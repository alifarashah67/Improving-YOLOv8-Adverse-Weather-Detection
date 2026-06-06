# 🌧️ Improving YOLOv8 Robustness for Adverse Weather Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00CFFF?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![Dataset](https://img.shields.io/badge/Dataset-BDD100K-orange?style=flat-square)](https://bdd-data.berkeley.edu/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> **A dataset-centric optimization pipeline for real-world object detection under rain, low-light, and adverse visibility conditions.**  
> Improves mAP50-95 from **0.188 → 0.304** (+61.7%) without architectural changes.

---

## 📌 Overview

Autonomous driving perception systems must remain reliable in challenging conditions — rain, nighttime, wet-road reflections, and reduced visibility. This project investigates a **practical, dataset-centric approach** to improve YOLOv8 robustness without designing a new architecture or requiring multimodal sensors.

The key insight: **most real-world performance problems are caused by dataset mismatch, not insufficient model complexity.**

---

## 🚀 Results

| Model | Change | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| YOLOv8s Baseline | Original class set | 0.540 | 0.289 | 0.283 | 0.188 |
| YOLOv8s Rain-only | Rain-focused augmentation | 0.493 | 0.356 | 0.306 | 0.214 |
| **YOLOv8m Clean + Rain** | Class filtering + rain focus + larger backbone | **0.540** | **0.443** | **0.432** | **0.304** |

### Per-Class AP (Final Model)
| Class | AP@0.5 |
|---|---|
| Drivable Area | 0.929 |
| Car | 0.460 |
| Traffic Light | 0.434 |
| Traffic Sign | 0.426 |
| Lane | 0.412 |
| Truck | 0.311 |
| Person | 0.299 |
| Bus | 0.183 |

---

## 🔬 Methodology

The pipeline follows four progressive steps:

```
YOLOv8s Baseline
      ↓
Weather Augmentation (rain-focused)
      ↓
Dataset Cleaning (remove underrepresented classes)
      ↓
Backbone Scaling (YOLOv8s → YOLOv8m)
```

### 1. Baseline Training
YOLOv8s trained on BDD100K-derived dataset with 12 original classes including bike, bus, car, drivable area, lane, motor, person, rider, traffic light, traffic sign, train, and truck.

### 2. Rain-Focused Specialization
Weather-specific augmentation targeting rain and dark/rain scenes. This alone improved mAP50-95 from 0.188 to 0.214.

### 3. Dataset Cleaning & Class Filtering
Severe class imbalance was observed — the validation split had only **3 train instances**, 33 motor instances, and 56 rider instances vs. thousands of car/lane objects. Removed unstable classes: `bike`, `motor`, `rider`, `train`.

**Final 8 classes:** `bus`, `car`, `drivable area`, `lane`, `person`, `traffic light`, `traffic sign`, `truck`

### 4. Backbone Scaling
Upgraded from YOLOv8s to YOLOv8m for increased representation capacity while maintaining real-time inference.

---

## 🗂️ Repository Structure

```
├── data/                   # Dataset configs and preprocessing scripts
├── notebooks/              # Training and evaluation notebooks
├── results/                # Metrics, confusion matrix, PR curves
├── predictions/            # Qualitative detection outputs
├── weights/                # Trained model weights
└── README.md
```

---

## ⚙️ Setup & Usage

### Requirements
```bash
pip install ultralytics torch torchvision
```

### Training
```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.train(
    data="data/bdd100k_rain.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0
)
```

### Inference
```python
model = YOLO("weights/best.pt")
results = model.predict("path/to/image.jpg", conf=0.25)
results[0].show()
```

---

## 📊 Training Details

| Parameter | Value |
|---|---|
| Base model | YOLOv8m |
| Input resolution | 640 × 640 |
| Epochs | 50 |
| Batch size | 16 |
| Hardware | NVIDIA A100 |
| Framework | PyTorch + CUDA |
| Dataset | BDD100K (rain subset) |

---

## 🔍 Qualitative Analysis

The optimized detector handles:
- ✅ Wet-road reflections
- ✅ Low-light / nighttime scenes  
- ✅ Rain and reduced visibility
- ✅ Drivable region and lane structure detection

Known failure cases: small/distant objects, heavily occluded road users in dense scenes.

---

## 📄 Paper

This repository accompanies the research paper:

> **"Improving YOLOv8 Robustness for Adverse Weather Road Scene Object Detection Using Dataset Cleaning and Weather-Focused Optimization"**  
> Ali Farashah — Independent Researcher, Berlin, Germany

---

## 🔮 Future Work

- Compare with YOLOv11 and RT-DETR
- Add ACDC or DAWN datasets for broader adverse-weather coverage
- Incorporate temporal video information and tracking
- Domain adaptation for generalization across weather conditions

---

## 📬 Contact

**Ali Farashah** — Computer Vision Engineer  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/alifarashah67)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/alifarashah67)
