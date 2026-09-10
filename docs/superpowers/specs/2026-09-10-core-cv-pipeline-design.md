# Core CV Pipeline — Design Spec

Date: 2026-09-10
Status: Approved, pending implementation plan

## Context

footsight's end goal: let a user freeze football (soccer) broadcast video, and
turn the resulting still into a 2D top-down pitch mockup for tactical
analysis/content creation.

That end goal spans several independent subsystems: video playback/freeze UI,
pitch calibration, player/ball detection, coordinate projection, and mockup
rendering/styling. This spec covers only the first sub-project: **a core CV
pipeline that takes a single still image and produces a rendered 2D mockup**,
run as a script against sample stills. This validates the hardest, most
uncertain part (calibration + detection accuracy) before any video-freeze UI
is built on top of it.

## Decisions carried in from feasibility spike

- No generative AI / paid image-generation APIs anywhere in this pipeline.
  Mockups are rendered deterministically in code from detected positions —
  zero per-image cost, fully local. (See `MEMORY.md`, 2026-09-10 entry.)
- Approach: pretrained models only, no custom training (Approach A from
  brainstorming). Fallback if pitch-calibration accuracy proves poor:
  classical CV (color segmentation + Hough line transform) for that one
  piece only — custom model training is out of scope unless both fail.

## Scope for this sub-project

- Language/stack: Python.
- Test images: user-provided sample broadcast stills (not a public dataset).
- Detection scope: players only, no team classification, no ball detection.
- Camera angle: main wide/tactical broadcast camera only. Close-ups and
  replays are out of scope — they don't have enough visible pitch geometry to
  calibrate a homography from.
- Output: a single rendered mockup image per input still. No UI, no batch/
  video processing yet.

## Architecture & components

A Python CLI script, not a service. Five independently testable modules:

- **`pitch_calibration`** — runs [PnLCalib](https://github.com/mguti97/PnLCalib)
  (pretrained single-view keypoint + line detection, SV_kp/SV_lines weights)
  on the still to compute a homography matrix mapping image pixels →
  real-world pitch coordinates (standard 105×68m pitch). PnLCalib is
  GPL-2.0 licensed, so it is vendored as a git submodule and run only as an
  isolated subprocess (a small glue script shells out to it and returns a
  JSON-encoded homography) — never imported in-process, so footsight's own
  code isn't bound by GPL-2.0 on distribution. See `MEMORY.md`, 2026-09-10.
- **`player_detection`** — runs torchvision's pretrained `fasterrcnn_resnet50_fpn`
  (COCO weights), filtered to the "person" label, returns bounding boxes. No
  training required. Imported in-process — torchvision's detection models are
  BSD-3-Clause, unlike Ultralytics YOLO (AGPL-3.0), so no subprocess isolation
  is needed here. See `MEMORY.md`, 2026-09-10.
- **`projection`** — takes each player's bounding box, uses the bottom-center
  point (the ground-contact point — a person's feet, not head/box-center,
  sit on the pitch plane) and applies the homography matrix to get real-world
  pitch coordinates. Pure function, fully unit-testable.
- **`render`** — draws a plain top-down pitch outline (placeholder style;
  visual polish/branding is a separate future decision, not part of this
  sub-project) and plots a dot per player at their projected position.
- **`pipeline`** (entry point) — wires the above together: image in →
  calibrate → detect → project → render → image out.

Dependencies: OpenCV (image I/O), `torch`/`torchvision` (player detection,
and shared with the PnLCalib subprocess environment), `scipy`/`pyyaml`/`tqdm`
(PnLCalib subprocess requirements), Pillow for rendering output.

## Data flow

```
still.jpg
   │
   ├─► pitch_calibration ──► homography matrix H
   │
   └─► player_detection ──► list of bounding boxes
                                  │
                                  ▼
                          projection (using H)
                                  │
                                  ▼
                    list of (x, y) pitch coordinates
                                  │
                                  ▼
                               render
                                  │
                                  ▼
                            mockup.png
```

Calibration and detection run independently off the same input still (no
dependency between them), then both feed into projection. Every stage's
output is a plain, inspectable data structure (matrix, list of boxes, list of
coordinates), so any stage's intermediate output can be dumped/printed/
plotted on its own during development to isolate where accuracy breaks down.

## Error handling

- **Calibration failure** (not enough keypoints found to compute a reliable
  homography) — fail loudly with a clear error message. Do not silently
  render positions from a bad/degenerate matrix. Expected to be rare given
  the main-wide-camera-only scope, but must be a hard stop, not a guess.
- **No players detected** — not an error. Render an empty pitch. Valid state
  (e.g. a still with no visible play).
- **Non-player detections** (crowd, bench, staff near the touchline) —
  filtered using the homography itself: project every detection, discard any
  point that falls well outside the real pitch boundary (105×68m + small
  margin). This is a free filter that falls out of the pipeline rather than
  a separate classification step.

## Testing

- `projection` — unit-testable with synthetic known inputs/outputs (pure
  homography math).
- `render` — unit-testable: given a known coordinate list, assert dots land
  at expected pixel positions.
- `pipeline` orchestration — testable with mocked calibration/detection
  outputs, to verify wiring without needing real models in CI.
- Calibration and detection accuracy are **not** unit-testable in the
  classic sense (model outputs, not deterministic logic). Validation is
  visual: run against user-provided sample stills and eyeball dot placement
  against the actual still. No automated accuracy metric for v1 — that would
  require hand-labeled ground truth, not justified yet for a feasibility
  check.

## Out of scope (explicitly, for this sub-project)

- Video playback, scrubbing, or freeze-frame capture UI.
- Team classification, ball detection/tracking.
- Any camera angle other than main wide/tactical broadcast.
- Mockup visual styling/branding — placeholder rendering only.
- Batch or video-frame-sequence processing.
- Custom model training.
