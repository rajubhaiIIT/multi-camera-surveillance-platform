# Results log

Every number in the main README comes from a row here. Rows marked **pending** are not measured yet.
Hardware for all runs: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), PyTorch 2.14 (CUDA 12.6), Ultralytics 8.4.171.

## Summary table

| Phase | Module | Dataset | Metric | Result | Reference | Date | Setup |
|---|---|---|---|---|---|---|---|
| 1 | Tracking (single camera) | MOT17 train, 7 sequences, persons | HOTA / IDF1 / MOTA | 41.4 / 48.2 / 42.5 (IDSW 914) | none (see notes) | 2026-10-03 | YOLO11n (COCO-pretrained, not fine-tuned) + ByteTrack, imgsz 1280, conf 0.1, FP16 |
| 1 | Speed | MOT17 train, 1080p | FPS (incl. image reading) | 32.2 | target 15+ | 2026-10-03 | same run, RTX 4060 Laptop |
| 1 | Detection | COCO val | mAP50-95 | pending | about 39.5 (published YOLO11n) | | |
| 2 | Person Re-ID | Market-1501 | Rank-1 / Rank-5 / mAP | 93.7 / 97.5 / 82.5 | roughly 94 / - / 85 (published ResNet50 baselines, approximate) | 2026-10-03 | ResNet50 + BNNeck, 512-d embedding, 60 epochs, ImageNet init |
| 2 | Person Re-ID, cross-dataset | Market-1501 to CUHK03 | Rank-1 / mAP | pending | | | |
| 2 | Multi-camera tracking | Own recording | cross-camera IDF1 / HOTA | pending | | | |
| 2 | Multi-camera tracking | WILDTRACK | cross-camera IDF1 / HOTA | pending | | | |

## Phase 1: tracking detail (run `yolo11n_1280`)

| Metric | Combined |
|---|---|
| HOTA | 41.4 |
| DetA (detection) | 42.4 |
| AssA (association) | 41.2 |
| IDF1 | 48.2 |
| MOTA | 42.5 |
| ID switches | 914 |
| False positives / false negatives | 14,840 / 48,866 |
| Tracker IDs created vs real people | 1,370 vs 546 |

Per-sequence HOTA: MOT17-02 31.0 | 04 45.4 | 05 35.8 | 09 46.8 | 10 36.2 | 11 49.2 | 13 36.6

Planned comparison runs (not done yet): `yolo11n_640`, `yolo11s_1280`.

## Phase 2: Re-ID detail (run `models/reid/resnet50_market`)

| Item | Value |
|---|---|
| Rank-1 / Rank-5 | 93.7 / 97.5 |
| mAP | 82.5 |
| Model | ResNet50, last stride 1, BNNeck, 512-d embedding |
| Training | Market-1501 train (12,936 images, 751 people), ID loss with label smoothing + batch-hard triplet loss, P=16 x K=4, Adam 3.5e-4 with 5-epoch warm-up and cosine decay, 60 epochs, mixed precision |
| Time | about 34 s per epoch, about 35 to 45 minutes in total |
| Suggested match threshold (cosine similarity) | _paste `threshold_suggestion` from `eval/runs/reid_market.json`_ |

## Notes and limitations (keep these in the README write-up)

- **The detector is not tuned for this data.** YOLO11n is COCO-pretrained and was never fine-tuned on MOT17 or CrowdHuman. Recall on MOT17 is about 56%, and missed people cap every tracking score. Scores are not comparable with published MOT17 results that use a detector trained on that data.
- **Single-camera numbers are not cross-camera numbers.** HOTA/IDF1/MOTA above score each camera on its own. Re-ID is scored with Rank-1/mAP, and multi-camera tracking with cross-camera IDF1. They are reported as separate rows.
- **Market-1501 is the easy test.** Training and test share the same six cameras. Expect lower scores on unseen cameras (the cross-dataset and own-recording rows).
- **The best checkpoint was chosen using the Market-1501 test set**, which is common practice for this benchmark but slightly optimistic.
- **Published reference numbers are quoted from memory and approximate.** Check them against the papers before using them in a final write-up.
- **Match threshold:** the value above is tuned on Market-1501 pairs only. Re-tune it on the own recording.
