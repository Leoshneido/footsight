# footsight

Turn a football (soccer) broadcast still into a 2D top-down pitch mockup.

## Setup

    git submodule update --init --recursive
    pip install -r requirements.txt
    pip install -e .
    mkdir -p weights/pnlcalib
    curl -L -o weights/pnlcalib/SV_kp https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp
    curl -L -o weights/pnlcalib/SV_lines https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines
    mkdir -p weights/roboflow
    pip install gdown
    gdown -O weights/roboflow/football-player-detection-v9.pt "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"
    mkdir -p weights/ultralytics
    curl -L -o weights/ultralytics/yolo11m-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt

The pose model (42.5 MB) is used for the camera-angle view.

The player-detection model comes from Roboflow's
[sports](https://github.com/roboflow/sports) repo (link taken from its
`examples/soccer/setup.sh`, which saves it as `football-player-detection.pt`).
The file is saved here as `-v9`; it is byte-identical to that download
(SHA-256 `75b09c377fbf9d0791d23f6cfb689f5aed6eaa43a6818bd1fb884cf7507fffaf`).

## Usage

    python -m footsight.pipeline path/to/still.jpg path/to/mockup.png

Each run writes two images: `mockup.png` (top-down) and `mockup_camera.png`
(the still's own camera angle, redrawn with a drawn stadium and players posed
like the real ones).

Options:

- `--pick-ball` -- review the ball on the still: the detected ball (if any)
  is circled; click to move it or add one, Delete/Backspace to remove it.
  Enter accepts, Esc keeps what detection found.
- `--remove-detections` -- shows every detected box on the still; click false
  ones (e.g. a broadcaster watermark) to remove them before the team split.
  Click again to restore. Enter accepts, Esc keeps everything.

## Capture studio (Chrome extension + live processing)

    python -m footsight.studio --session "Barca v Feyenoord"

Loads the models once, opens the editor, and waits for stills from the
footsight Chrome extension (see `extension/README.md` to load it): on the match
page press **⌘⇧S** and the frame is sent at full resolution, processed in the
background, and appears in the editor when ready (about 15 s). Everything for
the session goes to `captures/<date time> <name>/` (originals, both images,
analysis cache, fixes). Reopen a past session with
`python -m footsight.studio --open "captures/<folder>"`.

In the studio the editor also has fix tools, each redone in a few seconds
(and undoable):

| Key | Fix |
|---|---|
| X | Remove a wrong detection (watermark, ball boy, coach) |
| N | Add a missed player: click their feet, pick the side (1-4) |
| O | Click to place/move the ball; Shift+click removes it |
| V | Put a player on the right side (team, keeper, referee) |

## Overlay editor (live telestrator)

    python -m footsight.edit out/match/

Opens the camera views in that folder in your browser (served from your own
machine only), full screen, for drawing analysis graphics while you narrate
and screen-record. Graphics lie on the grass under the players.

| Key | Tool |
|---|---|
| H | Highlight a player (click again to remove) |
| T | Tag a player: type a name, Enter |
| A / P | Drag a curved run arrow / a straight dotted pass |
| K | Click players to link them, Enter to finish |
| L | Drag a line -- Shift: across the pitch, Alt: dashed |
| Z | Click points for a zone; click the first point again (or Enter) to close |
| S | Spotlight the highlighted players |
| 1-4 | Yellow, cyan, white, red |
| Cmd/Ctrl+Z, Shift+Cmd/Ctrl+Z | Undo, redo |
| C / Esc | Clear the still / cancel the shape in progress |
| Left / Right | Previous / next still |
| F | Full screen |
| B | Hide / show the toolbar (also the "Hide toolbar" box; the ☰ tab brings it back) |
| E / Cmd/Ctrl+S | Export `<still>_camera_edited.png` / save overlays to reopen later |

Nothing is written unless you export or save. Stills made before the editor
existed need the pipeline re-run (it now also writes the `_camera_background`,
`_camera_figures` and `_camera_scene.json` files the editor reads).

Tests: `pytest` for Python, `node --test footsight/editor/tests/*.test.mjs` for
the editor's geometry.

## Scope

Main wide/tactical broadcast camera stills only. Detects players,
goalkeepers and referees; splits outfield players into two teams by jersey
color; finds the ball (or you place it by hand); and draws everything on a
broadcast-style 2D pitch, with each team drawn in the shirt and shorts colors
read off the still.

Known limits: close-up and replay camera angles aren't supported, and
broadcast watermarks are sometimes detected as players -- remove them with
`--remove-detections`.

See
`docs/superpowers/specs/2026-09-10-core-cv-pipeline-design.md` for full
design rationale and `MEMORY.md` for the decisions behind it.
