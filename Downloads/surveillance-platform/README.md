# Intelligent Multi-Camera Surveillance Platform

A multi-camera person-tracking system that gives the same person the same ID across different cameras, raises alerts for events such as falls, fights and weapons, and lets an operator search the history through a web dashboard. Built entirely from free software and public datasets, with every result measured and reproducible.

## Quick start (Phase 0 infrastructure)

**Prerequisites:** Docker Desktop, Python 3.10+, ffmpeg, git, make (or use the PowerShell equivalents below).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env        # edit POSTGRES_PASSWORD before continuing
docker compose --env-file .env -f infra/docker-compose.yml up -d
python scripts/check_infra.py      # all four services must show [ OK ]
```

If OpenSearch fails on Linux or WSL2:
```bash
sudo sysctl -w vm.max_map_count=262144
```

---

## Phase 1: Detection and single-camera tracking

YOLO11n + ByteTrack, a worker that reads an RTSP camera, and MOT17 evaluation (HOTA / IDF1 / MOTA).

### Setup (GPU)

Install a CUDA build of PyTorch first, then the AI requirements:
```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-ai.txt
python scripts/check_gpu.py        # must show [ OK ] GPU: NVIDIA GeForce RTX 4060
```
If `lap` fails to install, change it to `lapx` in `requirements-ai.txt` and retry.

### Watch tracking on your fake camera

Make sure the camera container is running:
```powershell
docker compose --env-file .env -f infra/docker-compose.yml --profile cameras up -d
python -m ai.worker --camera-id cam1 --source rtsp://localhost:8554/cam1 --show
```
Press `q` to quit. The console prints `device=cuda:0` and an FPS counter.

### MOT17 evaluation
```powershell
python eval/run_mot17.py --run-name yolo11n_1280 --imgsz 1280
python eval/eval_mot17.py --run-name yolo11n_1280
```

### Phase 1 results

| Metric | Value | Setup |
|---|---|---|
| HOTA / IDF1 / MOTA | 41.4 / 48.2 / 42.5 (IDSW 914) | YOLO11n COCO-pretrained, imgsz 1280, conf 0.1, FP16 |
| Speed | 32.2 FPS | RTX 4060 Laptop, 1080p images including read |

---

## Phase 2: Person Re-ID and multi-camera tracking

### Part A: Train the Re-ID model on Market-1501

```powershell
# get Market-1501 from Kaggle or Hugging Face, save as data\downloads\market1501.zip, then:
python scripts/download_data.py fetch market1501

# quick 2-epoch trial first (about 1 minute)
python -m ai.reid.train --data data/raw/market1501 --epochs 2 --eval-every 1 --out models/reid/trial

# full training (about 35 minutes on RTX 4060)
python -m ai.reid.train --data data/raw/market1501 --out models/reid/resnet50_market

# evaluate
python -m ai.reid.eval_market --ckpt models/reid/resnet50_market/best.pt --data data/raw/market1501 --out eval/runs/reid_market.json

# export to ONNX (optional)
python -m ai.reid.export --ckpt models/reid/resnet50_market/best.pt --out models/reid/reid.onnx
```

### Part A results

| Dataset | Rank-1 | Rank-5 | mAP | Setup |
|---|---|---|---|---|
| Market-1501 (trained here) | 93.7% | 97.5% | 82.5% | ResNet50 + BNNeck, 512-d, 60 epochs, ImageNet init |
| CUHK03 cross-dataset (never seen) | 77.9% | 97.4% | 58.8% | same model, tested on CUHK03 new protocol (detected boxes) |

The 15.8-point drop from Market to CUHK03 is the **domain gap**: the model loses confidence when it moves to cameras and lighting it was never trained on. This directly explains why cross-camera matching on WILDTRACK is weak.

### Part A: database setup
```powershell
python scripts/init_db.py              # creates the two pgvector tables (safe to repeat)
pytest -m infra -k store               # database smoke test
```

### Part B: multi-camera recording tools

**Your own recording** (2-3 phones, 3-5 consenting friends, read `docs/recording_guide.md` first):
```powershell
python -m ai.recording sync  cam1=a.mp4 cam2=b.mp4 --out data/rec01/offsets.json
python -m ai.recording extract --video cam1=a.mp4 --video cam2=b.mp4 --offsets data/rec01/offsets.json --out data/rec01/run --max-seconds 30
# open data/rec01/run/sheets/, fill the person column of labels.csv, then:
python -m ai.recording score data/rec01/run --render
python -m ai.recording sweep data/rec01/run
```

**WILDTRACK public dataset** (7 cameras, ground truth included, no labeling, about 7 GB):
```powershell
python scripts/download_data.py fetch wildtrack
python -m ai.recording wildtrack --root data/raw/wildtrack/Wildtrack_dataset --out runs/wt --max-frames 50
python -m ai.recording score runs/wt --stale-after 6 --render
python -m ai.recording sweep runs/wt --stale-after 6
```

### Part B: diagnosis tools
```powershell
# detector only (no tracker, no Re-ID): AP50, AP50-95, recall by person size
python -m ai.recording detect-eval --root data/raw/wildtrack/Wildtrack_dataset --weights yolo11n.pt rtdetr-l.pt --imgsz 1280 --max-frames 50

# detection-only diagnosis from an existing run
python -m ai.recording diagnose runs/wt

# oracle: feed ground-truth boxes to test Re-ID and matching alone
python -m ai.recording wildtrack --root data/raw/wildtrack/Wildtrack_dataset --out runs/wt_oracle --max-frames 50 --oracle
python -m ai.recording score runs/wt_oracle --stale-after 6

# measure the Re-ID domain gap on WILDTRACK
python -m ai.recording export-reid --root data/raw/wildtrack/Wildtrack_dataset --out data/wt_reid
python -m ai.reid.eval_market --ckpt models/reid/resnet50_market/best.pt --data data/wt_reid
python -m ai.reid.train --data data/wt_reid --init-ckpt models/reid/resnet50_market/best.pt --lr 1e-4 --epochs 20 --out models/reid/resnet50_wt
```

### Part B results

| Experiment | IDF1 | HOTA | Key detail |
|---|---|---|---|
| WILDTRACK, 3 detectors (50 frames) | 8.6 – 11.0 | 9.6 – 10.9 | YOLO11n / YOLO11m / YOLOv8n-CrowdHuman, real detector |
| WILDTRACK oracle (perfect boxes) | 36.3 | 46.2 | Ground-truth boxes + per-camera IDs, tests Re-ID and matcher only |

**Detection only on WILDTRACK (AP50, imgsz 1280, no tracker):**

| Model | AP50 | AP50-95 | Recall | Precision |
|---|---|---|---|---|
| YOLO11n | 33.8 | 10.2 | 59.4% | 26.5% |
| YOLO11x | 32.3 | 9.6 | 61.7% | 25.6% |
| RT-DETR-L | 28.0 | 7.9 | 53.0% | 23.2% |

All three models are COCO-pretrained and score similarly, so switching COCO models is not the fix.

**What the evidence says:**
- Detection costs about 25 IDF1 points (from 36 with oracle to about 11 with real detectors).
- Even with perfect boxes the oracle reaches only 36.3 IDF1, so Re-ID and matching are also a bottleneck (AssA 21.6).
- The domain gap (Market → CUHK03 drop, and the tiny similarity gap in the threshold suggestion) explains why the matcher struggles on WILDTRACK.

---

## CUHK03 cross-dataset converter
```powershell
python scripts/convert_cuhk03.py --data data/raw/cuhk03 --out data/raw/cuhk03_market
python -m ai.reid.eval_market --ckpt models/reid/resnet50_market/best.pt --data data/raw/cuhk03_market
```

---

## Datasets

```powershell
python scripts/download_data.py list      # all datasets and how to get them
python scripts/download_data.py fetch mot17 lfw
python scripts/download_data.py verify
```

| Dataset | Used for | Method |
|---|---|---|
| MOT17 | Phase 1 tracking evaluation | Auto-download (~5 GB) |
| Market-1501 | Re-ID training and test | Manual: Kaggle/Hugging Face, save as data/downloads/market1501.zip |
| CUHK03 | Cross-dataset Re-ID test | Manual: Kaggle, save as data/downloads/cuhk03.zip |
| LFW | Face recognition (Phase 3) | Auto-download (~170 MB, link may have moved) |
| WILDTRACK | Multi-camera tracking | Auto-download (~7 GB, needs ~14 GB free while extracting) |

---

## Tests
```powershell
pytest                    # 99+ offline tests
pytest -m infra           # needs docker compose up
pytest -m model           # downloads weights, needs MOT17
pytest --basetemp=.pytest_tmp   # use if you see an access-denied error
```

Add `--basetemp=.pytest_tmp` to the `addopts` line in `pytest.ini` to make it permanent.

---

## Project layout

```
ai/                   detection, tracking, Re-ID, embedder, cross-camera matcher,
                      recording tools, WILDTRACK loader, diagnosis, sync
backend/              FastAPI app (Phase 7, not built yet)
frontend/             React dashboard (Phase 8, not built yet)
infra/                docker-compose, Postgres init SQL, MediaMTX, Nginx
eval/                 evaluation scripts and results
  results.md          every metric with its settings
  run_mot17.py        runs YOLO+ByteTrack on MOT17 sequences
  eval_mot17.py       scores the result with TrackEval
  eval_coco_detection.py   detection mAP on COCO val
scripts/              download_data.py, check_infra.py, check_gpu.py,
                      init_db.py, convert_cuhk03.py, fake_camera.sh
tests/                test_phase0.py, test_phase1.py, test_phase2.py,
                      test_phase2b.py, test_wildtrack.py
docs/                 recording_guide.md, ethics.md
data/
  downloads/          archive files (git-ignored)
  raw/                extracted datasets (git-ignored)
  sample_videos/      fake camera input (git-ignored)
models/
  reid/               .pt and .onnx checkpoints (git-ignored)
runs/                 recording and tracking run outputs (git-ignored)
```

---

## Planned phases (not built yet)

| Phase | Module |
|---|---|
| 3 | Face recognition (InsightFace, enrollment, LFW evaluation) |
| 4 | Pose estimation, fall detection, violence classifier |
| 5 | Weapon detection, vehicle tracking, ANPR |
| 6 | Image enhancement, stabilization, background subtraction, optical flow |
| 7 | FastAPI backend, JWT auth, alerts, WebSocket, OpenSearch |
| 8 | React dashboard: live grid, alerts, search, trajectories |
| 9 | Deployment: Nginx, ONNX/TensorRT, CI, one-command demo |

---

## Ethics, licenses and data

- Only public datasets and footage of consenting people are used.
- Face data is biometric personal data under India's DPDP Act 2023. Enrolled faces are stored only for the demo and can be deleted.
- Ultralytics YOLO is AGPL-3.0. InsightFace pretrained weights are non-commercial. Some datasets carry research-only terms.
- Record each dataset's license and date in `docs/ethics.md`.
- Videos and recordings stay out of git (`data/` and `runs/` are git-ignored).
- The CUHK03 and WILDTRACK results use the standard evaluation protocol. The Market-1501 best checkpoint was chosen using the test set, which is common practice but slightly optimistic.

---

## Hardware

All results were produced on an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), PyTorch 2.14 (CUDA 12.6), Ultralytics 8.4.171, Python 3.12, Windows 11.
