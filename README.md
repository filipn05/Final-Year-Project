# Quantifying Brand and Commercial Exposure in News Videos Using Computer Vision

Final Year Project — BSc Information Technology (Honours) (Artificial Intelligence)
Department of Computer Information Systems, Faculty of ICT, University of Malta

**Author:** Nikolina Filipov Pajić
**Supervisor:** Dr Dylan Seychell  **Co-supervisor:** Jonathan Attard

---

## Overview

This project investigates how effectively computer vision can be used to quantify brand exposure in news video broadcasts. A YOLO-based object detector is trained to recognise **26 Maltese brand logos** in TVM broadcast news footage, and the trained models are applied to full-length broadcasts to measure on-screen brand presence.

The work contributes the first annotated dataset of Maltese broadcast brands. None of the 26 brands appear in existing public logo datasets.

Two lightweight architectures, **YOLOv8n** and **YOLO11n**, are compared under identical training conditions.

## Pipeline

Training follows a two-phase approach:

1. **Phase 1 – class-agnostic pre-training.** Both architectures are pre-trained on [LogoDet-3K](https://github.com/Wangjing1551/LogoDet-3K-Dataset), with every logo mapped to a single class (`logo`), so the model first learns general logo appearance.
2. **Phase 2 – fine-tuning.** The pre-trained models are fine-tuned on the annotated Maltese dataset (26 brand classes).

The full workflow is:

```
TVM videos ──► frame extraction ──► annotation (Label Studio) ──► YOLO format
                                                                     │
LogoDet-3K ──► Phase 1 pre-training ──────────────┐                  ▼
                                                  ├──► Phase 2 fine-tuning ──► evaluation ──► video inference
                              train/val/test split + class balancing ┘
```

### Dataset design

- **1,323 annotated images**, ~3,000 bounding boxes, 26 brand classes.
- Images come from two sources: frames from TVM broadcast videos and promotional logo imagery collected online (added to address data scarcity).
- Online images are used for **training only**; validation and test sets contain **only TVM broadcast frames**.
- The primary split (**mixed_v2**) groups frames by **source video**, so near-duplicate frames from the same broadcast never appear across train and test.
- A naive **random split** is also trained for comparison; its inflated scores demonstrate the effect of near-duplicate leakage.
- Underrepresented classes are oversampled in the training set only, using offline augmentation.

## Repository structure

```
Final-Year-Project/
├── thesis-logodet3k/          # Phase 1: LogoDet-3K conversion and pre-training
├── thesis-maltese-dataset/    # Frame extraction, dataset preparation, Phase 2 fine-tuning,
│                              # evaluation and video inference
├── results/                   # Evaluation outputs (CSV)
├── figures/                   # Plots used in the thesis
├── requirements.txt
└── README.md
```

## Setup

All experiments were run in **Google Colab** on a **Tesla T4 GPU**, with data and checkpoints stored on Google Drive.

To run locally:

```bash
pip install -r requirements.txt
```

The Phase 1 notebooks download LogoDet-3K from Kaggle and require Kaggle API credentials, stored as Colab secrets (`KAGGLE_USERNAME`, `KAGGLE_KEY`).

Paths in the notebooks point to Google Drive (`/content/drive/MyDrive/...`). Update them to match your own setup.

## How to run

Run the notebooks in this order:

1. **Frame extraction** — extract frames from broadcast videos.
2. **Phase 1 pre-training** — convert LogoDet-3K to YOLO format and pre-train YOLOv8n and YOLO11n.
3. **Dataset preparation** — convert the Label Studio export to YOLO format, split the data and apply class balancing.
4. **Phase 2 fine-tuning** — fine-tune both models on the Maltese dataset.
5. **Evaluation** — compute test-set metrics, per-class results and confusion matrices.
6. **Video inference** — run the models over full broadcasts to measure brand exposure.

## Trained models

All trained weights are available in the [**v1.0 release**](https://github.com/filipn05/Final-Year-Project/releases/tag/v1.0):

- Phase 1 pre-trained models (LogoDet-3K, class-agnostic)
- Phase 2 fine-tuned models at 416 px (primary), 640 px and 1024 px

```python
from ultralytics import YOLO

model = YOLO("yolov8n_finetune_malta_mixed_v2_final.pt")
results = model.predict("frame.jpg", imgsz=416)
```

Use the same `imgsz` the model was trained at.

## Results

Test-set results on the mixed_v2 split (TVM broadcast frames only):

| Input size | Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| 416 px | YOLOv8n | 0.602 | 0.301 | 0.408 | 0.310 |
| 416 px | YOLO11n | 0.588 | 0.469 | 0.471 | 0.361 |
| 640 px | YOLOv8n | 0.686 | 0.552 | 0.624 | 0.497 |
| 640 px | YOLO11n | 0.831 | 0.497 | 0.586 | 0.398 |
| 1024 px | YOLOv8n | 0.592 | 0.644 | 0.648 | 0.517 |
| 1024 px | YOLO11n | 0.672 | 0.500 | 0.545 | 0.434 |

416 px is the primary configuration; 640 px and 1024 px form a resolution experiment. Full per-class results and discussion are in the thesis.

## Data availability

The annotated Maltese brand dataset, the source TVM broadcast footage, and the promotional images are **not publicly released**, as they contain third-party copyrighted material. Requests for access can be directed to Dr Dylan Seychell, Department of Computer Information Systems, University of Malta.

LogoDet-3K is publicly available from its original authors.

## Acknowledgements

- Dr Dylan Seychell (supervisor) and Jonathan Attard (co-supervisor), who provided the TVM recordings.
- The advertisement and news-story segment boundaries used to select footage come from Jonathan Attard's project *Queryable Computer Vision Analysis of News Videos*.
- LogoDet-3K: J. Wang, W. Min, S. Hou, S. Ma, Y. Zheng and S. Jiang, "LogoDet-3K: A Large-Scale Image Dataset for Logo Detection," *ACM Transactions on Multimedia Computing, Communications, and Applications*, 2022.
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
