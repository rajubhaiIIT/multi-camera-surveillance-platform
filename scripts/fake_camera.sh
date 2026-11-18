#!/usr/bin/env bash
# Publish a video file as a looping fake RTSP camera WITHOUT Docker for ffmpeg.
# Usage: scripts/fake_camera.sh <video> [camera_name]   e.g. scripts/fake_camera.sh data/sample_videos/sample.mp4 cam2
# Needs ffmpeg installed and MediaMTX running (make up).
set -euo pipefail
VIDEO="${1:?usage: fake_camera.sh <video> [camera_name]}"
NAME="${2:-cam1}"
PORT="${RTSP_PORT:-8554}"
exec ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 -i "$VIDEO" \
  -an -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p -g 30 \
  -f rtsp -rtsp_transport tcp "rtsp://localhost:${PORT}/${NAME}"
