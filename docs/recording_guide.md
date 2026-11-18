# Recording guide: your own multi-camera test

Goal: 2 or 3 phones film people walking between non-overlapping views, so you can test whether the system gives
the same person the same ID in every camera. This is the most convincing demo in the project.

## 1. People and consent (do this first)
- Film only friends who agree. Get a short written or chat message from each person, for example:
  *"I agree to be filmed for [your name]'s university/portfolio project on multi-camera tracking. The videos are
  stored privately, used only for this project, and deleted when it ends or whenever I ask."*
- Pick a place with few strangers (a quiet corridor, an empty classroom block, a courtyard on a holiday).
  Do not use footage where strangers are clearly identifiable.
- Never put the videos, crops, or contact sheets in your GitHub repo. `data/` is git-ignored; keep recordings there.
- Face and body images are personal data (DPDP Act 2023): note in `docs/ethics.md` who consented and when.

## 2. Equipment and setup
- 2 phones minimum, 3 is better. Landscape, 1080p, 30 fps.
- Fix each phone on a tripod, a shelf, or a stack of books. **No hand-held filming, no zoom.**
  Lock focus and exposure if your camera app allows it (long-press on the screen).
- Place the cameras so their views **do not overlap**: for example, one at each end of a corridor and one at a
  doorway or turning. Each view should show people at least 1.5 m tall in frame, from head to feet if possible.
- Note roughly how long it takes to walk from one view to the next (you can put this in the matcher config).
- Plug phones in or start with full batteries, and free up storage. 10 minutes of 1080p is a few GB per phone.

## 3. Time sync with a clap (every recording)
1. Start recording on **all phones**.
2. Stand near the phones and **clap once, loudly and sharply**, before anyone starts walking. One clap, then wait 3 seconds.
3. Stop all phones only after everyone is done.

`python -m ai.recording sync ...` finds the clap in each video's audio and works out the time offsets.
If it cannot find a clear clap, clap again louder or closer, or write the offsets by hand.

## 4. What to film (3 to 5 people, 5 to 10 minutes per camera)
Do these scenes in order. Every scene gets a name in your notes, with an approximate start time.

| Scene | What happens | Why |
|---|---|---|
| 1. Easy | One person at a time walks through camera 1, then 2, then 3 | Baseline |
| 2. Groups | Two or three people walk together through the cameras | Crowding and overlapping |
| 3. Crossing | Two people cross paths in front of a camera | ID switches |
| 4. Same clothes | Two friends in similar dark tops | The hardest honest case |
| 5. Clothes change | One person takes off a jacket between cameras | Where Re-ID is expected to fail |
| 6. Lighting | The same walk near a window, in shade, or with the lights switched | Lighting shift between cameras |

Keep some scenes simple, so you can see the system working, and keep the hard ones, so you can show honest failures.

## 5. Process it
```powershell
# A. sync (all the clip files, one clap each)
python -m ai.recording sync cam1=data/rec01/cam1.mp4 cam2=data/rec01/cam2.mp4 cam3=data/rec01/cam3.mp4 --out data/rec01/offsets.json

# B. detect + track + Re-ID (GPU). Try --max-seconds 30 first to check everything works
python -m ai.recording extract --video cam1=data/rec01/cam1.mp4 --video cam2=data/rec01/cam2.mp4 --video cam3=data/rec01/cam3.mp4 --offsets data/rec01/offsets.json --out data/rec01/run --max-seconds 30

# C. label who is who (see section 6), then score
python -m ai.recording score data/rec01/run --render
python -m ai.recording sweep data/rec01/run
```
Drop `--max-seconds` for the full run. `--stride 2` (the default) processes every second frame; use `--stride 3` if it is too slow.

## 6. Labeling (about 15 to 30 minutes)
Open `data/rec01/run/sheets/*.jpg`. Each row is one track: a caption (camera, track number, frames, times) and a few crops.
Open `labels.csv` in Excel or Notepad and fill the `person` column with a name for each track:

- **Same name for the same real person**, in every camera (for example `anna`).
- **Leave blank, or write `ignore`**, for strangers, false detections, and fragments you cannot identify.
- One person usually has several tracks in a camera (the tracker loses them behind others). Give all of them the same name.
- Do not edit the other columns. The labels file is never overwritten when you rerun `extract`'s sheets step, unless you pass `--force`.

The boxes come from the tracker, so the score measures **identity matching across cameras**, not detection quality.
Mention this in your write-up.

## 7. What to report
- `score`: IDF1 and HOTA across cameras, ID switches, identities found versus real people, cross-camera link precision/recall/F1.
- `sweep`: a threshold table, plus two ablations: appearance only (no timing/layout rules) and rules only (no Re-ID).
- Failure cases: save the rendered videos at the moments where it went wrong (similar clothes, clothes change). Honest failure analysis reads as maturity.
- Tune `sim_threshold` from the sweep, then say it was tuned on the same recording, or film a second short clip to test the chosen value.

## 8. Optional: camera layout file
Pass `--config topo.json` to `score` or `sweep` to tell the matcher about your layout:
```json
{"matcher": {"sim_threshold": 0.55, "max_gap_s": 300},
 "min_travel_s": {"cam1,cam2": 8, "cam2,cam3": 6},
 "overlapping": []}
```
`min_travel_s` is the shortest believable walk between two cameras. Only add it if you measured it.
