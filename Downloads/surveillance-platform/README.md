# Intelligent Multi-Camera Surveillance Platform

Phase 0: infrastructure, dataset access, and a fake RTSP camera.

## Prerequisites
- Docker Desktop (or Docker Engine + Compose v2)
- Python 3.10+
- `ffmpeg` on your PATH (only for making/probing sample videos)
- `make` (Windows: use WSL, or run the commands inside the Makefile directly)

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

make env            # creates .env  -> open it and change POSTGRES_PASSWORD
make up             # postgres+pgvector, redis, opensearch, mediamtx
make check          # every service must print [ OK ]
```

### OpenSearch fails to start? (Linux / WSL2)
OpenSearch needs a higher memory-map limit:
```bash
sudo sysctl -w vm.max_map_count=262144
```
On Docker Desktop with WSL2 run this inside the WSL distro (`wsl -d docker-desktop` if needed).
The compose file caps its heap at 512 MB, so a laptop with 8 GB RAM is enough.

## Datasets
```bash
python scripts/download_data.py list      # what exists and how to get it
make data                                 # auto-download + extract what it can
make data-verify                          # OK / MISSING per dataset
```
- **Automatic:** MOT17 and LFW.
- **Manual (no approval forms):** Market-1501, CUHK03, RWF-2000, UR Fall, WILDTRACK, a weapon set.
  Download in your browser, save the archive as `data/downloads/<name>.zip` (for example `market1501.zip`),
  then run `make data` again. It extracts and verifies for you.
- Links on research sites move. If an automatic download fails the script tells you and prints the manual steps.
- Record each dataset's license in `docs/ethics.md`.

## Fake camera
1. Make a video. Either drop any `.mp4` at `data/sample_videos/sample.mp4`, or build one from MOT17:
   ```bash
   make mot-video                 # uses MOT17-04-SDP; change with: make mot-video SEQ=MOT17-09-SDP
   ```
   (If your MOT17 folder layout differs, adjust the path in the Makefile.)
2. Start the camera:
   ```bash
   make up-cams
   make check                     # then:
   python scripts/check_infra.py --stream cam1
   ```
3. Watch it: `ffplay rtsp://localhost:8554/cam1` (or open it in VLC). Browser view: `http://localhost:8888/cam1`.

More cameras without Docker: `scripts/fake_camera.sh data/sample_videos/other.mp4 cam2`

## Tests
```bash
make test           # offline: config checks, dataset registry, safe extraction
make test-infra     # needs `make up` (and `make up-cams` + sample.mp4 for the stream test)
```

## Phase 0 exit checklist
- [ ] `make up` then `make check` shows 4x `[ OK ]` (Postgres reports pgvector)
- [ ] `python scripts/check_infra.py --stream cam1` succeeds, and the looped video plays in VLC/ffplay
- [ ] `make data-verify` shows at least MOT17 and Market-1501 as OK (the rest can finish during Phase 1-2)
- [ ] `make test` and `make test-infra` pass
- [ ] First commit pushed to GitHub (`.env` and `data/` are git-ignored)

## Layout
```
ai/ backend/ frontend/   filled in from Phase 1 onward
infra/                   docker-compose, MediaMTX, Postgres init, nginx (Phase 9)
scripts/                 download_data.py, check_infra.py, fake_camera.sh
eval/results.md          every metric you report
docs/ethics.md           licenses, consent, privacy
```

---

# Phase 1: Detection and single-camera tracking

YOLO11n + ByteTrack, a worker that reads an RTSP camera, and MOT17 evaluation (HOTA / IDF1 / MOTA).
Commands below are PowerShell; run them from the repo root with the virtual environment active.

## Phase 1 setup (GPU)
`pip` on Windows installs the **CPU-only** PyTorch by default, so install a CUDA build first.
Get the current command for your system from https://pytorch.org/get-started/locally/ (Stable, Windows, Pip, a CUDA version).
At the time of writing it looks like:
```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-ai.txt
python scripts/check_gpu.py        # must print [ OK ] GPU: NVIDIA GeForce RTX 4060 ...
```
`requirements-ai.txt` installs TrackEval from GitHub, so `git` must be installed (it is, from the earlier step).
If `check_gpu.py` says CUDA is not available, you got the CPU build: re-run the torch line above with `--force-reinstall`,
and update your NVIDIA driver if it still fails. Everything also works on CPU, just slower.

## 1. Watch tracking on your fake camera
Make sure the fake camera is running (the compose command with `--profile cameras` from Phase 0), then:
```powershell
python -m ai.worker --camera-id cam1 --source rtsp://localhost:8554/cam1 --show
```
A window shows boxes and track IDs with an FPS counter. Press `q` to quit.
`--show` needs the normal `opencv-python` package (not `-headless`), which `requirements-ai.txt` installs.
The first run downloads `yolo11n.pt` (about 6 MB).

Other useful options:
```powershell
# write the contract records to a file (one JSON object per line) and save an annotated video
python -m ai.worker --source rtsp://localhost:8554/cam1 --jsonl out.jsonl --save-video out.mp4 --max-frames 300
# run on a video file instead of a camera
python -m ai.worker --source data/sample_videos/sample.mp4 --show
```
Each line of `out.jsonl` looks like:
```json
{"camera_id": "cam1", "frame_ts": 1759400000.12, "frame_idx": 41, "track_id": 7, "bbox": [412.0, 220.5, 470.2, 380.1], "class": "person", "conf": 0.8731}
```
`track_id` is unique within one camera only. Cross-camera IDs arrive in Phase 2.

## 2. Measure it: MOT17 tracking metrics
```powershell
python eval/run_mot17.py --run-name yolo11n_1280 --imgsz 1280     # runs 7 sequences, writes result files
python eval/eval_mot17.py --run-name yolo11n_1280                 # HOTA, IDF1, MOTA, ID switches
python eval/run_mot17.py --run-name yolo11n_640 --imgsz 640       # a second run to compare speed vs accuracy
python eval/eval_mot17.py --run-name yolo11n_640
```
Each `eval_mot17.py` run prints a ready-made row; paste it into `eval/results.md`.
Results and run settings are saved under `eval/runs/<run-name>/` (`metrics.json`, `run_meta.json`).

Optional detection benchmark (downloads COCO val, about 1 GB): `python eval/eval_coco_detection.py`

## 3. Tests
```powershell
pytest                  # offline: contract, MOT format, worker, TrackEval scoring checks
pytest -m model         # real YOLO on real frames (downloads yolo11n.pt; needs MOT17 for the full check)
pytest -m infra         # services + fake camera from Phase 0
```

## Phase 1 exit checklist
- [ ] `check_gpu.py` shows your RTX 4060
- [ ] `python -m ai.worker ... --show` displays stable IDs on the fake camera
- [ ] `eval/runs/yolo11n_1280/metrics.json` exists and the row is in `eval/results.md`
- [ ] Worker FPS on one stream is 15+ (the console prints it; the MOT17 run also reports FPS)
- [ ] `pytest` and `pytest -m model` pass
- [ ] Committed and pushed
