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

## Phase 3: Face Recognition and Identification

InsightFace `buffalo_l` (SCRFD face detector + ArcFace ResNet50 embedder), in-memory cosine enrollment index, temporal majority-voting track pipeline, and LFW verification evaluation.

### Usage

```powershell
# 1. Enroll people from photos into the index
python -m ai.face.enroll enroll --id alice --name "Alice Smith" --photos data/faces/alice/
python -m ai.face.enroll list

# 2. Search query photo against the index
python -m ai.face.enroll search --photo query.jpg

# 3. Benchmark on LFW (10-fold cross validation)
python -m ai.face.eval_lfw --pairs data/raw/lfw/pairs.txt --lfw-root data/raw/lfw/lfw --pick center --tar-at-far 0.01 0.001 0.0001

# 4. Run face recognition across multi-camera tracking videos
python -m ai.recording face-run runs/epfl_terrace_rtdetr --index models/face_index.json --render
```

### Phase 3 results

| Benchmark / Condition | Metric | Value | Detail |
|---|---|---|---|
| **LFW Standard Verification** | 10-fold CV Accuracy | **99.817 ± 0.189%** | Best threshold 0.301, 100% detection rate |
| **LFW Security ROC** | TAR @ FAR = 1% | **99.800%** | Standard operational surveillance threshold |
| **LFW Security ROC** | TAR @ FAR = 0.1% | **99.733%** | High-security access threshold |
| **LFW Security ROC** | TAR @ FAR = 0.01% | **99.700%** | Extreme low false-alarm threshold (AUC = 0.99941) |
| **CCTV Degradation (0.5x Scale)** | 10-fold CV Accuracy | **99.850 ± 0.174%** | Downsampled to 125x125 px |
| **CCTV Degradation (0.25x Scale)** | 10-fold CV Accuracy | **99.767 ± 0.327%** | Downsampled to ~62x62 px |
| **CCTV JPEG Artifacts (Quality 30)** | 10-fold CV Accuracy | **99.833 ± 0.167%** | Heavy compression noise |

---

## Phase 4: Pose Estimation, Fall Detection & Violence Classification

Real-time human pose estimation (YOLO11n-Pose with 17 COCO keypoints), physics-based temporal fall detection, and multi-person violent struggle classification.

### Features

- **17-Point COCO Pose Extraction:** Shoulders, elbows, wrists, hips, knees, ankles, and facial points with confidence filtering.
- **Biomechanical Fall Engine:**
  - Torso vector inclination monitoring ($\theta_{torso} > 60^\circ$ relative to vertical).
  - Body aspect ratio inversion ($w / h > 0.95$).
  - Vertical descent velocity profiling ($v_y > 1.0$ body heights/sec).
  - Post-fall dwell timer (distinguishes genuine falls from sitting, bending, or tying shoes).
  - Recovery state tracking (`upright` -> `falling` -> `fallen` -> `recovered`).
- **Violent Interaction Detector:** Pairwise spatial proximity and arm kinematic energy tracking to detect punches, strikes, and physical struggles.
- **Full Video / Camera Stream Visualization:** Skeletons, bounding boxes, posture angles, and alert overlays rendered directly onto footage.

### Usage

```powershell
# 1. Run live pose and fall detection on any video file or stream
python -m ai.pose.runner demo --source data/sample_videos/walk.mp4 --save-video out.mp4

# 2. Run pose, fall and violence detection on an existing multi-camera run
python -m ai.pose.runner run-eval runs/epfl_terrace_rtdetr --render
```

### Phase 4 results

| Task / Anomaly | Dataset / Benchmark | Metric | Value | Detail |
|---|---|---|---|---|
| **Fall Detection** | UR Fall Detection (URFD), 10 sequences | Sensitivity / Specificity / F1 | **80.0% / 60.0% / 72.7%** | YOLO11n-Pose + Biomechanical state machine, 3.37s latency, 95.9 FPS |
| **Violence / Fight** | TNUE Surveillance CCTV, 10 sequences | Sensitivity / Specificity / F1 | **100.0% / 100.0% / 100.0%** | YOLO11n-Pose + Spatio-temporal energy engine, 2.24s latency, 58.5 FPS |


---

## Phase 5: Weapon Detection, Vehicle Tracking & ANPR

Detects physical security threats (weapons/threat objects), tracks vehicles across traffic/parking cameras, and recognizes vehicle license plates with hotlist/watchlist alerting.

### Features

- **Weapon Detection & Threat Association:**
  - Detects knives, firearms, bats, and scissors.
  - Geometry-aware person association: checks weapon bounding box containment and proximity to hands/body of tracked people to determine if a weapon is actively held/brandished.
  - Real-time `WeaponAlert` generation with visual warnings.
- **Multi-Vehicle Tracking & Traffic Monitoring:**
  - Tracks cars, motorcycles, buses, trucks, and bicycles.
  - Trajectory speed estimation and illegal parking / loitering detection ($T_{stationary} \ge 30$s).
- **Automatic Number Plate Recognition (ANPR):**
  - Morphological top-hat + Sobel edge localization targeting rectangular license plates ($2.0 \le w/h \le 5.5$).
  - EasyOCR character recognition with CLAHE contrast enhancement.
  - Format cleaning and validation (Indian HSRP formats `[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}` and generic formats).
  - Hotlist / Watchlist index (`PlateIndex`) with exact and fuzzy Levenshtein edit distance matching (recovering OCR slips like '0' vs 'D' or '1' vs 'I').

### Usage

```powershell
# 1. Manage vehicle watchlist (hotlist / stolen vehicles)
python -m ai.vehicles.runner watchlist add --plate MH12DE1433 --reason "Stolen Vehicle" --owner "John Doe"
python -m ai.vehicles.runner watchlist list

# 2. Run vehicle tracking & ANPR on video/camera
python -m ai.vehicles.runner demo --source data/sample_videos/traffic.mp4 --watchlist models/vehicle_watchlist.json --save-video out_vehicles.mp4
```

---

## Phase 6: Image Enhancement, Stabilization, Background Subtraction & Optical Flow

Algorithms for handling low-light, camera vibration, moving object detection, and scene velocity fields in challenging CCTV environments.

### Features

- **Low-Light & Contrast Enhancement (`ai.enhancement.enhance`):**
  - **CLAHE:** Contrast Limited Adaptive Histogram Equalization applied strictly on the Lightness ($L$) channel of the LAB color space to prevent chromatic noise or color shift.
  - **Gamma Correction:** Non-linear power-law mapping ($I_{out} = I_{in}^{1/\gamma}$) to illuminate deep surveillance shadows.
  - **Unsharp Masking:** High-frequency edge boosting for de-blurring license plates, faces, and apparel.
- **Video Stabilization (`ai.enhancement.stabilizer`):**
  - Lucas-Kanade sparse feature optical flow tracking up to 200 corner points across frames.
  - Estimates rigid 2D camera motion (translation $\Delta x, \Delta y$, rotation $\Delta \theta$).
  - Exponential moving window trajectory smoothing to cancel wind shake and pole vibrations without losing deliberate camera panning.
- **MOG2 Background Subtraction & Motion Activity (`ai.enhancement.motion`):**
  - Gaussian Mixture Model background subtraction with shadow filtering.
  - Morphological opening and dilation for sensor noise suppression.
  - Real-time **Motion Activity % meter** and bounding box tracking around dynamic regions.
- **Dense Optical Flow (`ai.enhancement.flow`):**
  - Gunnar Farneback dense optical flow computing motion vectors $(dx, dy)$ per pixel.
  - **Arrow Vector Grid:** Overlays yellow directional velocity arrows on motion centroids.
  - **HSV Motion Heatmap:** Color represents direction of motion; brightness represents velocity (useful for detecting sudden crowd panic or stampedes).

### Usage

```powershell
# 1. Low-light enhancement with side-by-side comparison
python -m ai.enhancement.runner enhance --source data/sample_videos/sample.mp4 --mode low_light --compare --show

# 2. Camera shake stabilization with side-by-side comparison
python -m ai.enhancement.runner stabilize --source data/sample_videos/sample.mp4 --smoothing 15 --compare --show

# 3. Background subtraction & motion detection
python -m ai.enhancement.runner motion --source data/sample_videos/sample.mp4 --show

# 4. Dense optical flow vector fields
python -m ai.enhancement.runner flow --source data/sample_videos/sample.mp4 --mode arrows --show
```

---

## Phase 7: FastAPI Backend, JWT Auth, Alerts, WebSocket & OpenSearch

Production-ready RESTful and real-time backend service providing authentication, camera registry, event ingestion, alert notifications, and search.

### Features

- **Authentication & RBAC (`backend.auth`):**
  - Secure bcrypt password hashing with PyJWT token lifecycle (`HS256`).
  - Role-Based Access Control guards: `admin` (full management), `operator` (cameras & alerts), `viewer` (read-only audit).
  - Built-in demonstration accounts (`admin` / `admin123`, `operator` / `operator123`, `viewer` / `viewer123`).
- **Camera Registry & Streaming (`backend.routes.cameras`):**
  - Dynamic camera registration, configuration, and health monitoring.
  - Multi-protocol stream URL resolution (RTSP, WebRTC, and HLS) via MediaMTX.
- **Alert Ingestion & Acknowledgment (`backend.routes.alerts`):**
  - Immediate alert dispatch with severity classifications (`critical`, `high`, `medium`, `low`, `info`).
  - One-click acknowledgment endpoint with operator audit notes.
- **Real-Time WebSocket Streaming (`backend.ws`, `backend.routes.ws`):**
  - `/ws/live` and `/ws/alerts` channels with pub/sub topic filtering (`camera:{id}`, `alerts`, `all`).
  - Instantaneous push notification of falls, fights, weapons, and watchlist matches.
- **OpenSearch & Hybrid Storage Fallback (`backend.opensearch_service`, `backend.store`):**
  - Distributed full-text search across events and alerts via OpenSearch.
  - Zero-dependency fallback to persistent in-memory indexing when external Docker services are offline, enabling 100% offline testability.

### Usage

```powershell
# 1. Start the FastAPI backend server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# 2. Access interactive Swagger API documentation
# Open your browser to: http://127.0.0.1:8000/docs
```

---

## Phase 8: React Operator Dashboard

A real-time command center interface connecting to the FastAPI backend and live WebSocket streams.

### Features

- **Operations Dashboard (`frontend/src/pages/DashboardPage.jsx`):**
  - Live system overview: cameras online, active threat counts, and real-time event counters.
  - Active alerts feed and live infrastructure service health monitors (PostgreSQL, Redis, OpenSearch, MediaMTX).
- **Camera Fleet Grid (`frontend/src/pages/CamerasPage.jsx`):**
  - Multi-camera video cards with animated live status indicators, RTSP URLs, and location metadata.
  - Modal form for registering new IP cameras.
- **Alert Triage & Acknowledgment (`frontend/src/pages/AlertsPage.jsx`):**
  - Multi-filter threat triage (severity, alert type, resolved/unresolved).
  - Operator acknowledgment dialog with audit notes.
- **Forensic Search & Trajectories (`frontend/src/pages/EventsPage.jsx`):**
  - Multi-camera log exploration: filter by camera ID, event type (tracklets, face, pose, weapon, vehicle), or global person ID.
- **Vehicle Watchlist & ANPR Hotlist (`frontend/src/pages/WatchlistPage.jsx`):**
  - Stolen/flagged license plate registry with high-security embossed plate badges.
  - Sighting history showing detected vehicle speed and camera location.
- **Real-Time WebSocket Feed (`frontend/src/useWebSocket.js`):**
  - Low-latency auto-reconnecting client on `/ws/live` delivering instantaneous slide-in toast notifications when critical threats occur.

### Usage

```powershell
# 1. Start the FastAPI backend (Terminal 1)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# 2. Start the React dashboard (Terminal 2)
npm run dev --prefix frontend

# 3. Open your browser:
# Navigate to: http://localhost:3000
# Quick login credentials: admin / admin123 or operator / operator123
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
| LFW | Face recognition (Phase 3) | Auto-download (~170 MB) |
| WILDTRACK | Multi-camera tracking | Auto-download (~7 GB, needs ~14 GB free while extracting) |

---

## Tests
```powershell
pytest                    # 187+ offline tests (Phase 0, 1, 2, 2b, 3, 4, 5, 6, 7, 8, WILDTRACK)
pytest -m infra           # needs docker compose up
pytest -m model           # downloads weights, needs GPU
pytest --basetemp=.pytest_tmp   # use if you see an access-denied error
```

---

## Project layout

```
ai/                   detection, tracking, Re-ID, face recognition, pose, vehicles, weapons, enhancement,
                      recording tools, WILDTRACK loader, diagnosis, sync
  enhancement/        CLAHE, gamma, unsharp mask, video stabilizer, MOG2 motion, dense optical flow
  face/               detector (SCRFD + ArcFace), index, pipeline, eval_lfw, enroll CLI
  pose/               YOLO-Pose detector, fall detector, violence detector, pipeline, runner CLI
  weapons/            threat & weapon detector, person-holding association, alert schema
  vehicles/           vehicle tracker, ANPR plate reader, watchlist index, pipeline, runner CLI
backend/              FastAPI production backend, JWT auth, RBAC, camera registry, alerts, WebSocket, OpenSearch
  routes/             auth, cameras, alerts, events, vehicles, stats, ws
  auth.py             password hashing, JWT verification, role dependencies
  config.py           system environment configuration
  main.py             FastAPI application & lifespan
  opensearch_service.py  OpenSearch client & full-text indexing
  schemas.py          Pydantic validation schemas
  store.py            unified state store with PostgreSQL/OpenSearch fallbacks
  ws.py               real-time WebSocket connection & broadcast manager
frontend/             React 19 + Vite operator dashboard, Lucide icons, live WebSockets
  src/                App.jsx, Layout.jsx, api.js, useWebSocket.js, index.css
  src/pages/          DashboardPage.jsx, CamerasPage.jsx, AlertsPage.jsx, EventsPage.jsx, WatchlistPage.jsx, LoginPage.jsx
  dist/               production build bundle (HTML/CSS/JS)
  vite.config.js      reverse proxy to FastAPI on port 8000
infra/                docker-compose, Postgres init SQL, MediaMTX, Nginx
eval/                 evaluation scripts and results
  results.md          every metric with its settings
  run_mot17.py        runs YOLO+ByteTrack on MOT17 sequences
  eval_mot17.py       scores the result with TrackEval
  eval_coco_detection.py   detection mAP on COCO val
scripts/              download_data.py, check_infra.py, check_gpu.py,
                      init_db.py, convert_cuhk03.py, fake_camera.sh
tests/                test_phase0.py, test_phase1.py, test_phase2.py,
                      test_phase2b.py, test_phase3.py, test_phase4.py, test_phase5.py,
                      test_phase6.py, test_phase7.py, test_phase8.py, test_wildtrack.py
docs/                 recording_guide.md, ethics.md
data/
  downloads/          archive files (git-ignored)
  raw/                extracted datasets (git-ignored)
  sample_videos/      fake camera input (git-ignored)
models/
  reid/               .pt and .onnx checkpoints (git-ignored)
  insightface/        buffalo_l face models (git-ignored)
runs/                 recording and tracking run outputs (git-ignored)
```

---

## Planned phases (upcoming)

| Phase | Module | Status |
|---|---|---|
| 1 | Detection and single-camera tracking (YOLO11n + ByteTrack, MOT17) | **Completed** |
| 2 | Person Re-ID and multi-camera tracking (Market-1501, WILDTRACK, EPFL) | **Completed** |
| 3 | Face recognition (InsightFace buffalo_l, enrollment, LFW, CCTV robustness) | **Completed** |
| 4 | Pose estimation, fall detection, violence classifier | **Completed** |
| 5 | Weapon detection, vehicle tracking, ANPR | **Completed** |
| 6 | Image enhancement, stabilization, background subtraction, optical flow | **Completed** |
| 7 | FastAPI backend, JWT auth, alerts, WebSocket, OpenSearch | **Completed** |
| 8 | React dashboard: live grid, alerts, search, trajectories | **Completed** |
| 9 | Deployment: Nginx, ONNX/TensorRT, CI, one-command demo | **In Progress / Next** |

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
