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
