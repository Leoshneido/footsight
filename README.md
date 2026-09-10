# footsight

Turn a football (soccer) broadcast still into a 2D top-down pitch mockup.

## Setup

    git submodule update --init --recursive
    pip install -r requirements.txt
    curl -L -o weights/pnlcalib/SV_kp https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp
    curl -L -o weights/pnlcalib/SV_lines https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines

## Usage

    python -m footsight.pipeline path/to/still.jpg path/to/mockup.png

## Scope

Main wide/tactical broadcast camera stills only. Detects players (no team
split, no ball) and plots them on a plain 2D pitch outline. See
`docs/superpowers/specs/2026-09-10-core-cv-pipeline-design.md` for full
design rationale and `MEMORY.md` for the decisions behind it.
