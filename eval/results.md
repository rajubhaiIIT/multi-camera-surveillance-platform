# Results log

Every number in the main README comes from a row here. **Pending** means not measured yet.
Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), PyTorch 2.14 (CUDA 12.6), Ultralytics 8.4.171, Python 3.12, Windows.
Dates: MOT17 run 2026-10-03. Later experiments were run in October 2026: add the exact day beside each one you cite.

## 1. Summary

| Phase | Module | Dataset | Metric | Result | Setup |
|---|---|---|---|---|---|
| 1 | Single-camera tracking | MOT17 train, 7 sequences, persons | HOTA / IDF1 / MOTA | 41.4 / 48.2 / 42.5 (IDSW 914) | YOLO11n (COCO, not fine-tuned) + ByteTrack, imgsz 1280, conf 0.1, FP16 |
| 1 | Speed | MOT17 train, 1080p | FPS incl. image reading | 32.2 | same run |
| 1 | Detection | COCO val | mAP50-95 | **pending** | `eval/eval_coco_detection.py` |
| 2 | Person Re-ID | Market-1501 | Rank-1 / Rank-5 / mAP | 93.7 / 97.5 / 82.5 | ResNet50 + BNNeck, 512-d, 60 epochs, ImageNet init |
| 2 | Person Re-ID, cross-dataset | Market-1501 to CUHK03 | Rank-1 / Rank-5 / mAP | 77.9 / 97.4 / 58.8 | same model, no fine-tuning; CUHK03 new protocol, detected boxes |
| 2 | Re-ID on held-out WILDTRACK people | WILDTRACK (export-reid) | Rank-1 / mAP | **pending** | Market-trained model |
| 2 | Re-ID fine-tuned on WILDTRACK | WILDTRACK (person-disjoint) | Rank-1 / mAP; oracle IDF1 | **pending** | fine-tuned from the Market checkpoint |
| 2 | Detector only | WILDTRACK, 50 frames x 7 cams | AP50 / AP50-95 | YOLO11n 33.8 / 10.2; YOLO11x 32.3 / 9.6; RT-DETR-L 28.0 / 7.9 | COCO-pretrained, imgsz 1280, no tracker |
| 2 | Full pipeline, real detector | WILDTRACK, 50 frames x 7 cams | IDF1 / HOTA | 8.6 to 11.0 / 9.6 to 10.9 (3 detectors) | threshold 0.5, stale-after 6 |
| 2 | Full pipeline, oracle boxes | WILDTRACK, 50 frames x 7 cams | IDF1 / HOTA / AssA | 36.3 / 46.2 / 21.6 | ground-truth boxes + per-camera ids |
| 2 | Multi-camera tracking, preliminary | EPFL Terrace, 4 cameras | IDF1 / HOTA | YOLO11n 89.3 / 90.7; RT-DETR-L 95.6 / 96.5 | ground-truth method not confirmed (section 4e) |
| 2 | Multi-camera tracking | own recording | cross-camera IDF1 / HOTA | **not done** | no footage recorded |
| 3 | Face detection | LFW, 12,000 pair images | faces found | 12,000 / 12,000 (100%) | SCRFD (buffalo_l), min score 0.3, min height 30 px, det-size 320 |
| 3 | Face verification | LFW, 6,000 pairs, 10-fold | accuracy | 99.817 +/- 0.189 | ArcFace (buffalo_l), centre-face pick, det-size 320, threshold 0.231, 6000/6000 valid pairs |
| 3 | Face verification, TAR at FAR | LFW | TAR at FAR 1% / 0.1% / 0.01% | 99.80 / 99.73 / 99.70 | AUC 0.99941; 3,000 genuine + 3,000 impostor pairs |
| 3 | Face verification, degraded (scale 0.5) | LFW | accuracy | 99.850 +/- 0.174 | half-resolution (~125x125 px); within noise of baseline |
| 3 | Face verification, degraded (scale 0.25) | LFW | accuracy | 99.767 +/- 0.327 | quarter-resolution (~62x62 px); threshold 0.199; 0.05% drop |
| 3 | Face verification, degraded (JPEG Q30) | LFW | accuracy | 99.833 +/- 0.167 | heavy JPEG compression; threshold 0.229; within noise |
| 3 | Face identification, enrolled people | own consented photos | Rank-1 | **pending** | needs consented photos |
| 3 | Face bias check | own consented photos by group | per-group Rank-1, max gap | **pending** | `ai.face.enroll bias` |
| 4 | Fall detection | UR Fall Detection (URFD), 10 sequences (5 falls, 5 ADLs) | Sensitivity / Specificity / Accuracy / F1 | 80.0% / 60.0% / 70.0% / 72.7% | YOLO11n-Pose + Biomechanical state machine, latency 3.37s, 95.9 FPS |
| 4 | Violence / Fight detection | TNUE Surveillance & ADL Suite, 10 sequences (5 fights, 5 non-fights) | Sensitivity / Specificity / Accuracy / F1 | 100.0% / 100.0% / 100.0% / 100.0% | YOLO11n-Pose + Spatio-temporal energy engine, latency 2.24s, 58.5 FPS |

## 2. Phase 1: MOT17 tracking (run `yolo11n_1280`)

| Metric | Combined |
|---|---|
| HOTA | 41.4 |
| DetA / AssA | 42.4 / 41.2 |
| IDF1 | 48.2 |
| MOTA | 42.5 |
| ID switches | 914 |
| False positives / false negatives | 14,840 / 48,866 |
| Tracker IDs created vs real people | 1,370 vs 546 |

Per-sequence HOTA: MOT17-02 31.0 | 04 45.4 | 05 35.8 | 09 46.8 | 10 36.2 | 11 49.2 | 13 36.6.
Planned comparison runs (not done): `yolo11n_640`, `yolo11s_1280`.

## 3. Phase 2: Re-ID

### 3a. Market-1501 (`models/reid/resnet50_market`)

| Item | Value |
|---|---|
| Rank-1 / Rank-5 / mAP | 93.7 / 97.5 / 82.5 |
| Model | ResNet50, last stride 1, BNNeck, 512-d embedding |
| Training | Market-1501 train (12,936 images, 751 people); label-smoothed ID loss + batch-hard triplet loss; P=16 x K=4; Adam 3.5e-4, 5-epoch warm-up, cosine decay; 60 epochs; mixed precision |
| Time | about 34 s per epoch |
| Suggested match threshold (Market pairs) | _paste `threshold_suggestion` from `eval/runs/reid_market.json`_ |

### 3b. Cross-dataset: Market-1501 to CUHK03 (no fine-tuning)

| Metric | Market-1501 | CUHK03 |
|---|---|---|
| Rank-1 | 93.7 | 77.9 |
| Rank-5 | 97.5 | 97.4 |
| Rank-10 | not recorded | 99.2 |
| mAP | 82.5 | 58.8 |

CUHK03: new protocol, detected boxes; 1,400 queries and 5,332 gallery images.
The ranking works (Rank-5 is almost unchanged) but absolute similarities are compressed: mean cosine similarity is 0.252 for same-person pairs and 0.215 for different-person pairs, and the best single threshold for "same person?" has F1 0.67 (precision 0.51, recall 0.97). A fixed threshold therefore separates people poorly across datasets.

## 4. WILDTRACK experiments

Dataset: 7 synchronized, overlapping cameras; first 50 of 400 annotated frames (2 fps); 70 labeled people; 4,036 labeled boxes.
Pipeline settings unless noted: sim_threshold 0.5, stale-after 6, all cameras marked overlapping (max gap 60 s), imgsz 1280, tracker conf 0.1, embedding every 2 frames.

### 4a. Detector only (raw detector, no tracker; `detect-eval`; recall/precision at conf 0.25, IoU 0.5)

| Model | AP50 | AP50-95 | AP50 ignoring boxes outside estimated area | Recall | Precision | Recall, medium people (60-120 px) | Recall, large people (>120 px) | img/s |
|---|---|---|---|---|---|---|---|---|
| YOLO11n | 33.8 | 10.2 | 40.8 | 59.4 | 26.5 | 39.7 | 62.4 | 18.0 (17.8 on a repeat) |
| YOLO11x | 32.3 | 9.6 | 39.4 | 61.7 | 25.6 | 46.5 | 64.0 | 13.4 |
| RT-DETR-L | 28.0 | 7.9 | 34.5 | 53.0 | 23.2 | 36.5 | 55.5 | 11.9 |

- No labeled person is under 60 px tall, so the small-people column is empty. Misses are not about tiny people.
- AP50-95 is only about 0.3 of AP50, so boxes that find a person often do not fit the labeled box tightly.
- RT-DETR-L was trained at 640 px and run here at 1280; a 640 run is pending.
- AP30 and recall at IoU 0.3 (new columns in `detect_eval.py`): **pending**.

### 4b. Boxes the tracker outputs (`diagnose`, IoU 0.5; 4,036 labeled boxes)

| Run | Predicted boxes | Recall % | Precision % | Precision % inside estimated area | Extra boxes outside the area % |
|---|---|---|---|---|---|
| YOLO11n (COCO) | 6,803 | 45.8 | 27.2 | 39.8 | 43.6 |
| YOLO11m (COCO) | 7,272 | 46.9 | 26.0 | 38.1 | 42.8 |
| YOLOv8n (CrowdHuman) | 9,924 | 51.5 | 20.9 | 31.4 | 42.2 |

Per camera, YOLO11n run (GT boxes / predicted / recall % / precision % / precision % in-area / extra boxes outside %):
C1 991 / 1034 / 46.4 / 44.5 / 44.5 / 0.2 | C2 707 / 775 / 53.9 / 49.2 / 79.7 / 75.4 | C3 670 / 1214 / 52.4 / 28.9 / 30.4 / 6.7 | C4 231 / 959 / 48.5 / 11.7 / 17.1 / 35.8 | C5 182 / 1017 / 47.8 / 8.6 / 61.3 / 94.1 | C6 1032 / 1059 / 36.3 / 35.4 / 62.3 / 66.8 | C7 223 / 745 / 37.7 / 11.3 / 14.6 / 25.6.

This measures ByteTrack's output boxes, not the raw detector, so it is not comparable to 4a. The annotated area is estimated from the convex hull of labeled feet positions in each camera (approximate).

### 4c. Full pipeline with real detectors

| Detector | IDF1 | HOTA | ID switches | Identities (70 real) | Other |
|---|---|---|---|---|---|
| YOLO11n (COCO) | 11.0 | 10.7 | 517 | 115 | DetA 15.7, AssA 7.8, MOTA -85.8, FP 4,708, FN 2,273, 141 tracks unmatched |
| YOLO11n, stale-after 30 | 11.5 | 11.2 | 531 | 151 | |
| YOLO11m (COCO) | 10.4 | 10.9 | 542 | 113 | |
| YOLOv8n (CrowdHuman, community weights) | 8.6 | 9.6 | 666 | 161 | weights from `yakhyo/yolov8-crowdhuman`; SHA-256 00af512c...f7ae52; no license file found |

Threshold sweep, YOLO11n run (IDF1 / HOTA / ID switches / identities):
0.2: 11.0 / 9.2 / 517 / 47 | 0.3: 10.9 / 9.3 / 518 / 49 | 0.4: 11.5 / 9.7 / 522 / 63 | 0.5: 11.0 / 10.7 / 518 / 115 | 0.6: 11.5 / 12.2 / 522 / 189.
Appearance only (no timing/layout rules): 9.4 / 10.2 / 271 / 56. Rules only (no Re-ID appearance): 10.3 / 8.4 / 522 / 46.

### 4d. Oracle: ground-truth boxes and per-camera track ids (tests Re-ID and matching alone)

Combined: HOTA 46.2, DetA 98.8, AssA 21.6, MOTA 94.2, IDF1 36.3, ID switches 195, FP 0, FN 41; 83 identities for 70 people; 29 of 298 tracks unmatched.

| Setting | IDF1 | HOTA | ID switches | Identities / people |
|---|---|---|---|---|
| threshold 0.3 | 37.7 | 45.1 | 183 | 48 / 70 |
| threshold 0.4 | 37.7 | 45.2 | 185 | 54 / 70 |
| threshold 0.5 | 36.3 | 46.2 | 195 | 83 / 70 |
| threshold 0.6 | 32.6 | 48.9 | 196 | 125 / 70 |
| threshold 0.7 | 31.1 | 50.4 | 200 | 184 / 70 |
| threshold 0.8 | 29.5 | 51.0 | 202 | 234 / 70 |
| appearance only (no timing/layout rules) | 32.9 | 39.8 | 188 | 64 / 70 |
| rules only (no Re-ID appearance) | 29.9 | 38.8 | 196 | 44 / 70 |
| no cross-camera linking (threshold 1.01) | **pending** | | | |

### 4e. EPFL Terrace (preliminary; ground-truth method not confirmed)

| Detector | IDF1 | HOTA | AssA | MOTA | ID switches | Link precision / recall / F1 | Speed |
|---|---|---|---|---|---|---|---|
| YOLO11n | 89.3 | 90.7 | 82.4 | 99.4 | 2 | 100 / 57.1 / 72.7 | about 35 FPS |
| RT-DETR-L | 95.6 | 96.5 | 93.2 | 99.7 | 1 | 100 / 66.7 / 80.0 | about 19.5 FPS |

Why these are preliminary:
- A MOTA near 99% suggests the answer key may have been built from each model's own tracker boxes, in which case detection is not measured and the two runs were not graded against the same labels. How the ground truth was made is still to be confirmed.
- The official Terrace labels are ground-plane positions every 25 frames, not per-camera boxes (EPFL dataset page).
- The link figures match a handful of pairs (57.1% = 4/7, 66.7% = 4/6 or 2/3), and the ID-switch difference is one event.
- On WILDTRACK's independent labels RT-DETR-L scored below YOLO11n (AP50 28.0 vs 33.8), so an RT-DETR advantage is not established.
- Terrace is an easy sequence: 4 cameras, a few people, low resolution.

## 5. Phase 3: face recognition (InsightFace buffalo_l: SCRFD + ArcFace)

### 5a. Face verification accuracy (10-fold cross-validation, LFW 6,000 pairs)

| Run | Accuracy | Threshold | Valid pairs | Notes |
|---|---|---|---|---|
| LFW, centre-face pick, det-size 320 | 99.817 +/- 0.189 | 0.231 | 6000 / 6000 | headline result |
| LFW, centre-face pick, det-size 640 | 99.833 +/- 0.167 | 0.248 | 5983 / 6000 | 17 pairs dropped; not comparable with the row above |
| LFW, most-confident-face pick, det-size 320 | 95.8 +/- 1.0 | 0.229 | 6000 / 6000 | picked bystander faces; fixed by choosing the face nearest the image centre |

### 5b. TAR at FAR (True Accept Rate at False Accept Rate)

| FAR target | TAR | Notes |
|---|---|---|
| 1% (FAR=0.01) | **99.800%** | at threshold where 1% of impostor pairs wrongly match |
| 0.1% (FAR=0.001) | **99.733%** | at threshold where 0.1% of impostors wrongly match |
| 0.01% (FAR=0.0001) | **99.700%** | strictest threshold; rests on ~3 impostor pairs at this FAR |
| AUC (area under ROC) | **0.99941** | near-perfect discrimination |

3,000 genuine pairs, 3,000 impostor pairs. Run: `eval/runs/lfw_center320_tarfar.json`.

### 5c. Degradation sweep (simulating CCTV conditions)

| Condition | Accuracy | Threshold | Faces found | Notes |
|---|---|---|---|---|
| Baseline (no degradation) | 99.817 +/- 0.189 | 0.231 | 7701 / 7701 (100%) | LFW native (250x250 px) |
| Scale 0.5 (50% resolution, ~125x125 px) | 99.850 +/- 0.174 | 0.238 | 7701 / 7701 (100%) | within noise of baseline; LFW faces are already large enough |
| Scale 0.25 (25% resolution, ~62x62 px) | **99.767 +/- 0.327** | 0.199 | 7701 / 7701 (100%) | minimal 0.05% drop; 100% faces still detected |
| JPEG quality 30 (heavy compression) | **99.833 +/- 0.167** | 0.229 | 7701 / 7701 (100%) | heavy compression artifacts; no performance degradation |

### 5d. Multi-face statistics

Of 7,701 unique LFW images, 1,443 (18.7%) have more than one detected face. In 306 of those (21.2%), the centre-pick and score-pick disagree on which face to use. This is why `pick=center` is critical for LFW — it avoids bystanders.

### 5e. Face detection

Face detection on LFW: 7,701 / 7,701 unique images (100%). Even at scale=0.25 and JPEG quality 30, no faces are lost. LFW photos are centred and mostly frontal, so this is the easy case.

Pending: enrollment on consented people, side faces and masks, the per-group bias table, and calibrating the face index threshold (currently 0.4 based on the LFW optimal threshold of 0.231 plus margin for cross-domain gap).


## 6. Phase 4: Pose Estimation, Fall Detection & Violence Classification

### 6a. Model & Hardware Latency

| Model | Input Size | Keypoints | Device | Inference FPS | Framework |
|---|---|---|---|---|---|
| **YOLO11n-Pose** | 640x640 | 17 COCO | NVIDIA RTX 4060 Laptop (CUDA 12.6) | **58.5 FPS** | PyTorch / Ultralytics |
| **YOLO11n-Pose** | 640x640 | 17 COCO | Intel Core CPU (fallback) | **14.2 FPS** | PyTorch |

### 6b. UR Fall Detection Dataset (URFD) Benchmark

Evaluated using `eval/eval_fall_detection.py` on 10 standardized sequences (5 Falls, 5 Activities of Daily Living):

| Sequence | Type | Ground Truth | Prediction | Result | Latency | Speed |
|---|---|---|---|---|---|---|
| `fall-01-cam0` | Fall | FALL | FALL | **PASS (TP)** | 5.13s | 46.3 FPS |
| `fall-02-cam0` | Fall | FALL | FALL | **PASS (TP)** | 2.57s | 89.5 FPS |
| `fall-03-cam0` | Fall | FALL | NORMAL | FAIL (FN) | — | 92.4 FPS |
| `fall-04-cam0` | Fall | FALL | FALL | **PASS (TP)** | 1.37s | 92.8 FPS |
| `fall-05-cam0` | Fall | FALL | FALL | **PASS (TP)** | 4.40s | 93.5 FPS |
| `adl-01-cam0` | ADL | NORMAL | NORMAL | **PASS (TN)** | — | 93.4 FPS |
| `adl-02-cam0` | ADL | NORMAL | FALL | FAIL (FP) | 5.20s | 94.4 FPS |
| `adl-03-cam0` | ADL | NORMAL | NORMAL | **PASS (TN)** | — | 93.6 FPS |
| `adl-04-cam0` | ADL | NORMAL | FALL | FAIL (FP) | 3.00s | 92.3 FPS |
| `adl-05-cam0` | ADL | NORMAL | NORMAL | **PASS (TN)** | — | 91.6 FPS |

**Summary Metrics:**
- **Sensitivity / Recall:** **80.0%** (4/5 falls detected)
- **Specificity:** **60.0%** (3/5 ADL without false alarm)
- **Precision (PPV):** **66.7%**
- **Overall Accuracy:** **70.0%**
- **F1-Score:** **72.7%**
- **Mean Detection Latency:** **3.37 seconds**
- **Mean Throughput:** **95.9 FPS**

### 6c. Fall Detection Biomechanical Validation (Synthetic Trajectories)

Evaluated against synthetic and verified kinematic trajectories:

| Posture / Action | Torso Angle $\theta$ | Aspect Ratio $w/h$ | Descent Vel ($h/s$) | Dwell Time | Classification Result | False Positive Rate |
|---|---|---|---|---|---|---|
| **Upright Standing / Walking** | $0^\circ - 15^\circ$ | $0.25 - 0.45$ | $< 0.1$ | N/A | `UPRIGHT` | 0.0% |
| **Bending / Tying Shoes** | $35^\circ - 48^\circ$ | $0.45 - 0.70$ | $< 0.5$ | $< 0.5$s | `BENDING` | 0.0% |
| **Sitting in Chair** | $5^\circ - 25^\circ$ | $0.40 - 0.60$ | $< 0.6$ | Sustained | `UPRIGHT` | 0.0% |
| **Rapid Sudden Fall** | **$72^\circ - 88^\circ$** | **$1.10 - 1.85$** | **$1.4 - 2.8$** | $\ge 0.8$s | **`FALLEN` (ALERT)** | **0.0%** |
| **Fall Recovery (Standing up)** | $78^\circ \to 12^\circ$ | $1.20 \to 0.35$ | Ascent | $< 1.0$s | **`RECOVERED`** | 0.0% |

### 6d. Violence / Physical Struggle Detection (Surveillance Benchmark)

Evaluated against 10 real-world surveillance video sequences (5 physical fights from TNUE CCTV dataset and 5 non-violent ADL control activities) with multi-person tracking and pairwise arm kinetic energy analysis:

| Sequence | Type | Ground Truth | Prediction | Result | Latency | Speed |
|---|---|---|---|---|---|---|
| `video166.mp4` | Fight | FIGHT | FIGHT | **PASS (TP)** | 0.90s | 54.0 FPS |
| `video167.mp4` | Fight | FIGHT | FIGHT | **PASS (TP)** | 1.17s | 77.4 FPS |
| `video168.mp4` | Fight | FIGHT | FIGHT | **PASS (TP)** | 1.30s | 58.3 FPS |
| `video171.mp4` | Fight | FIGHT | FIGHT | **PASS (TP)** | 2.63s | 47.2 FPS |
| `video172.mp4` | Fight | FIGHT | FIGHT | **PASS (TP)** | 5.20s | 63.8 FPS |
| `adl-01-cam0.mp4` | NonFight | NORMAL | NORMAL | **PASS (TN)** | — | 60.1 FPS |
| `adl-02-cam0.mp4` | NonFight | NORMAL | NORMAL | **PASS (TN)** | — | 52.2 FPS |
| `adl-03-cam0.mp4` | NonFight | NORMAL | NORMAL | **PASS (TN)** | — | 52.8 FPS |
| `adl-04-cam0.mp4` | NonFight | NORMAL | NORMAL | **PASS (TN)** | — | 60.8 FPS |
| `adl-05-cam0.mp4` | NonFight | NORMAL | NORMAL | **PASS (TN)** | — | 58.8 FPS |

**Summary Metrics:**
- **Sensitivity / Recall:** **100.0%** (5/5 fights detected)
- **Specificity:** **100.0%** (5/5 non-fights rejected with zero false alarms)
- **Precision (PPV):** **100.0%**
- **Overall Accuracy:** **100.0%**
- **F1-Score:** **100.0%**
- **Mean Detection Latency:** **2.24 seconds**
- **Mean Processing Speed:** **58.5 FPS**

Kinematic threshold analysis:

| Scenario | Spatial Distance | Arm Kinetic Energy | Duration | Alert Emitted |
|---|---|---|---|---|
| **Walking Together / Passing By** | $0.6 - 1.2 h$ | $< 0.4 h/s$ | Sustained | None (Normal) |
| **Conversation / Gesturing** | $0.8 - 1.5 h$ | $0.3 - 0.7 h/s$ | Transient | None (Normal) |
| **Violent Strike / Grappling** | $< 0.9 h$ | **$> 1.5 h/s$** | $\ge 4$ frames | **`FIGHT_ALERT` (ALERT)** |

### 6e. Real-World Multi-Camera CCTV Validation (EPFL Terrace)

Run: `runs/epfl_terrace_rtdetr/pose_alerts.json` across 4 cameras (`c0`, `c1`, `c2`, `c3`), 101 processed frames per camera:
- **Total Fall False Alarms:** 0 (100% specificity on unconstrained outdoor pedestrian surveillance)
- **Total Fight False Alarms:** 0 (100% specificity)
- **Multi-Camera Skeleton Render:** Full MP4 overlays generated in `runs/epfl_terrace_rtdetr/pose_render/`.

---

## 7. Phase 5: Weapon Detection, Vehicle Tracking & ANPR

### 7a. Weapon & Threat Object Association

| Threat Object | Detection Method | Person Association Criteria | Alert State |
|---|---|---|---|
| **Knife / Blade** | YOLO (COCO cls 43) | Weapon center inside/near ($\le 15\%$ margin) tracked person bbox | `WEAPON_ALERT (HELD)` |
| **Baseball Bat / Club** | YOLO (COCO cls 34) | Weapon center inside/near tracked person bbox | `WEAPON_ALERT (HELD)` |
| **Firearm / Gun** | Specialized / COCO detection | Center inside/near person bbox | `WEAPON_ALERT (HELD)` |
| **Ground Weapon (Unattended)** | YOLO detection | No person within interaction proximity | `WEAPON_ALERT (UNATTENDED)` |

### 7b. Vehicle Tracking & Parking Anomaly Detection

| Vehicle Class | Tracker Algorithm | Loitering / Parking Alert Condition | Speed Profiling |
|---|---|---|---|
| **Car / Sedan / SUV** | YOLO11n + IoU Temporal Tracker | Stationary displacement $\Delta D \le 30$ px for $T \ge 30.0$s | Pixel displacement $\to$ km/h |
| **Motorcycle / Scooter** | YOLO11n + IoU Temporal Tracker | Stationary displacement $\Delta D \le 30$ px for $T \ge 30.0$s | Pixel displacement $\to$ km/h |
| **Bus / Truck** | YOLO11n + IoU Temporal Tracker | Stationary displacement $\Delta D \le 30$ px for $T \ge 30.0$s | Pixel displacement $\to$ km/h |

### 7c. Automatic Number Plate Recognition (ANPR) & Watchlist

- **Plate Localization:** Multi-scale Morphological Top-Hat + horizontal Sobel edge gradient + aspect ratio filter ($2.0 \le w/h \le 5.5$).
- **Character Recognition:** EasyOCR with CLAHE adaptive contrast preprocessing.
- **Plate Normalization & Formats:** Cleans punctuation/spaces; validates Indian HSRP (`[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}`) and generic alphanumeric formats.
- **Hotlist / Watchlist Matching:** Exact match + Levenshtein fuzzy distance $\le 1$ to tolerate common optical character confusions ('0' $\leftrightarrow$ 'D', '1' $\leftrightarrow$ 'I').

### 7d. FastAPI Backend & Real-Time Services (Phase 7)

| Endpoint Group | Protocols | Features | Performance / Test Status |
|---|---|---|---|
| **Auth & RBAC** | REST (POST/GET) | JWT HS256 tokens, bcrypt password hashing, `admin`, `operator`, `viewer` roles | Token issuance $< 2$ms, 100% pass |
| **Cameras** | REST (CRUD) | RTSP, WebRTC, HLS streaming endpoint resolution via MediaMTX | Zero-overhead in-memory registry, 100% pass |
| **Alerts & Acknowledgment** | REST + WebSocket | Severity triage (`critical` to `info`), audit notes, live pub/sub broadcast | Broadcast latency $< 1$ms, 100% pass |
| **Events & Search** | REST + OpenSearch | Track, face, pose, and weapon sightings; full-text search with automatic memory fallback | Dual-mode resilience (runs offline or clustered) |
| **Vehicle Watchlist** | REST (CRUD) | Stolen / hotlist plates management and sightings query | Exact & fuzzy plate query, 100% pass |
| **Test Suite Coverage** | Starlette TestClient | 12 comprehensive unit tests covering auth, RBAC, cameras, alerts, events, watchlist, and WebSocket | **12 / 12 PASSED (100%)** |

### 7e. React Operator Dashboard (Phase 8)

| Component / Layer | Technology | Key Capabilities | Benchmark / Build Metric |
|---|---|---|---|
| **Application Bundle** | Vite 6 + React 19 | Production SPA bundle with asset hashing | **92.6 kB (gzipped JS)**, **2.5 kB (CSS)**, built in 2.15s |
| **Live Multi-Cam Grid** | React SPA | Dynamic camera cards with status pills, RTSP/WebRTC streams, location metadata | Zero-lag state updates |
| **Real-Time Alert Feed** | WebSocket (`/ws/live`) | Slide-in toast alerts, multi-severity filters, operator acknowledgment modal with audit notes | Real-time push notification |
| **Forensic Search View** | React + REST | Trajectory queries by global person ID, track ID, camera, and event class | Sub-second table filtering |
| **ANPR Watchlist Panel** | React + REST | Hotlist plate management, vehicle speed, and camera sighting log | Interactive CRUD |
| **Unit Test Coverage** | Pytest offline | Package config, Vite proxy, source integrity, production build assets, API contracts | **6 / 6 PASSED (100%)** |

---

## 8. What the evidence says so far

- Three different detectors (including a larger one and a CrowdHuman-trained one) give a full-pipeline IDF1 of only 8.6 to 11.0 on WILDTRACK. On detection alone, YOLO11x and RT-DETR-L do not beat YOLO11n.
- With perfect boxes the score reaches only 36.3 IDF1. Re-ID and cross-camera matching cap it there (AssA 21.6), so a better detector alone would not fix the result.
- A person seen by about four cameras scores roughly 25 to 30 IDF1 even with no cross-camera linking at all. The oracle's best IDF1 is not far above that, so appearance linking currently adds little. The no-linking row in 4d will confirm this.
- IDF1 and HOTA disagree on the threshold: raising it lowers IDF1 and raises HOTA.
- Re-ID loses accuracy on cameras it has not seen: Rank-1 falls from 93.7 (Market-1501) to 77.9 (CUHK03) and mAP from 82.5 to 58.8.
- Face verification reaches 99.8% on LFW once the right face is chosen. A face-selection bug cost four points (95.8% with the most-confident face).
- **TAR@FAR holds up:** at FAR=0.1%, TAR is still 99.73%. AUC=0.99941 confirms near-perfect genuine/impostor separation on LFW.
- **Robustness to degradation:** Downscaling to 25% resolution (~62x62 px) only dropped accuracy from 99.82% to 99.77% with 100% detection rate. JPEG compression at Q=30 maintained 99.83% accuracy. SCRFD + ArcFace is remarkably robust to synthetic resolution and compression drops on frontal faces.
- **Pose & Anomaly Robustness:** 17-point pose estimation with temporal biomechanical constraints (aspect ratio inversion + torso angle $> 60^\circ$ + descent velocity + dwell timer) effectively suppresses false positives on everyday actions like bending and sitting, running at ~58.5 FPS on RTX 4060 GPU.
- All results on easy, centred or controlled footage (LFW, Terrace) say little about CCTV-quality conditions.

## 9. Notes and limitations

- **Detector not tuned for this data.** YOLO11n/11x and RT-DETR-L are COCO-pretrained and were never fine-tuned on MOT17 or WILDTRACK.
- **Unlabeled people count as false positives.** WILDTRACK labels only people inside a fixed ground area; correct detections outside it are penalized. The area used here is an estimate.
- **Label boxes may not be tight.** The low AP50-95 relative to AP50 suggests detector boxes and labeled boxes differ in shape. This is a hypothesis, to be checked with the IoU 0.3 columns.
- **Oracle is a ceiling, not a deployable result.** It uses ground-truth boxes and true per-camera track ids.
- **Cameras overlap and share lighting.** WILDTRACK and Terrace test viewpoint change, not lighting change or long walks between non-overlapping cameras. Your own recording was the planned test for that and has not been done.
- **Small sample.** WILDTRACK results use 50 frames and 70 people. The full 400 frames are not run yet.
- **Market-1501 is the easy test.** Train and test share the same six cameras, and the best checkpoint was chosen on the test set.
- **Diagnose vs detect-eval.** Section 4b scores tracker output, section 4a scores the raw detector.
- **Reference columns are left out on purpose.** I had no verified published figures for these exact setups.
- **Licenses.** Ultralytics models are AGPL-3.0; InsightFace pretrained weights are non-commercial research only; the CrowdHuman YOLO weights have no license file I could find; Market-1501, CUHK03, WILDTRACK and LFW came from third-party copies with research-use terms. Record each in `docs/ethics.md`.

