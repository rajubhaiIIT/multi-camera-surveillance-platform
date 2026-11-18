# Data, Privacy, and Licensing Notes

- Only public datasets and footage of individuals who gave explicit consent are used.
- **Biometric Personal Data Compliance (India DPDP Act 2023 & GDPR):** Face embeddings and biometric facial identifiers constitute sensitive personal data. Enrolled face embeddings are stored locally in the face index solely for real-time security verification with consent, are decoupled from plain biometric images, and can be removed at any time via `python -m ai.face.enroll remove --id <person_id>`.
- Known model licenses:
  - Ultralytics YOLO (YOLO11 / ByteTrack): AGPL-3.0.
  - InsightFace buffalo_l (SCRFD + ArcFace ResNet50): Non-commercial research use only.
  - PyTorch & Torchvision: BSD-3-Clause.
  - ONNX Runtime: MIT License.

## Datasets Used

| Dataset | Source | License / Terms | Downloaded | Purpose in Platform |
|---|---|---|---|---|
| **MOT17** | MOTChallenge / CVPR 2016 | Academic / Non-commercial research | 2026-10-05 | Single-camera multi-object tracking benchmark (HOTA / IDF1 / MOTA) |
| **Market-1501** | Tsinghua University / UTS | Academic research only | 2026-10-05 | Re-ID backbone training (ResNet50 + BNNeck) and evaluation |
| **CUHK03** | Chinese University of Hong Kong | Academic research only | 2026-10-18 | Cross-dataset Re-ID domain gap evaluation |
| **WILDTRACK** | EPFL / ETH Zurich | Academic research only | 2026-10-05 | Multi-camera 7-view real-world tracking & occlusion benchmarking |
| **EPFL Terrace** | EPFL Computer Vision Lab | Academic research only | 2026-10-05 | Multi-camera tracking, temporal calibration & Re-ID evaluation |
| **LFW (Labeled Faces in the Wild)** | UMass Amherst | Academic research / Public domain | 2026-10-18 | Face verification accuracy, CCTV resolution degradation & ROC benchmarking |

## Demographic Bias & Fairness Protocol

The face verification pipeline provides a built-in demographic fairness evaluation utility:
```bash
python -m ai.face.enroll bias --data <path_to_grouped_faces> --out eval/runs/face_bias.json
```
- **Evaluation criteria:** Measures per-subgroup Rank-1 accuracy and cosine similarity separation across demographic groups (gender, age, ethnicity) to ensure false rejection rate (FRR) disparity remains within acceptable thresholds ($\le 2\%$) before operational deployment.
- **CCTV Degradation Safeguards:** As verified in `eval/results.md`, the platform enforces a minimum face height threshold (default: 30–40 px) and high detector confidence (`min_det_score=0.4`) to prevent false matches on severely blurred or distant subjects.
