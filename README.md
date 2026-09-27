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

The player-detection model comes from Roboflow's
[sports](https://github.com/roboflow/sports) repo (link taken from its
`examples/soccer/setup.sh`, which saves it as `football-player-detection.pt`).
The local copy was named `-v9`; not yet confirmed to be the identical file.

## Usage

    python -m footsight.pipeline path/to/still.jpg path/to/mockup.png

Options:

- `--pick-ball` -- review the ball on the still: the detected ball (if any)
  is circled; click to move it or add one, Delete/Backspace to remove it.
  Enter accepts, Esc keeps what detection found.
- `--remove-detections` -- shows every detected box on the still; click false
  ones (e.g. a broadcaster watermark) to remove them before the team split.
  Click again to restore. Enter accepts, Esc keeps everything.

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
