# Core CV Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Given one broadcast still image of a soccer pitch, produce a rendered 2D top-down mockup showing player positions — a CLI pipeline, no UI.

**Architecture:** Two independent pretrained-model stages (pitch calibration via PnLCalib, player detection via torchvision Faster R-CNN) feed a pure-function projection stage, which feeds a rendering stage. Calibration runs in an isolated subprocess (GPL-2.0 boundary); everything else runs in-process.

**Tech Stack:** Python, OpenCV, PyTorch/torchvision, Pillow, pytest. PnLCalib (vendored git submodule, GPL-2.0, subprocess-only).

**Spec:** `docs/superpowers/specs/2026-09-10-core-cv-pipeline-design.md`

## Global Constraints

- No generative AI / paid image-generation APIs anywhere in this pipeline — everything runs locally, zero per-image cost.
- Pretrained models only — no custom training.
- Camera angle: main wide/tactical broadcast camera only.
- Detection scope for this sub-project: players only — no team classification, no ball detection.
- PnLCalib must never be imported in-process — it is GPL-2.0 licensed and runs only as an isolated subprocess.
- Player detection uses torchvision (BSD-3-Clause), not Ultralytics YOLO (AGPL-3.0).

---

### Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `footsight/__init__.py`
- Create: `.gitignore`

**Interfaces:**
- Produces: the `footsight` package that every later task adds modules to.

- [ ] **Step 1: Create the package directory and files**

```bash
mkdir -p footsight tests tests/fixtures scripts weights/pnlcalib
touch footsight/__init__.py
```

- [ ] **Step 2: Write `requirements.txt`**

```
opencv-python
numpy
Pillow
torch
torchvision
scipy
pyyaml
tqdm
pytest
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
venv/
weights/
stills/
out/
*.DS_Store
```

- [ ] **Step 4: Install dependencies and verify the package imports**

```bash
pip install -r requirements.txt
python -c "import footsight"
```

Expected: no output, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt footsight/__init__.py .gitignore
git commit -m "chore: scaffold footsight package"
```

---

### Task 2: `projection` module

**Files:**
- Create: `footsight/projection.py`
- Test: `tests/test_projection.py`

**Interfaces:**
- Consumes: nothing (pure functions, no dependency on earlier tasks).
- Produces:
  - `project_point(H: np.ndarray, point: tuple[float, float]) -> tuple[float, float]`
  - `project_points(H: np.ndarray, points: list[tuple[float, float]]) -> list[tuple[float, float]]`
  - `bbox_to_ground_point(bbox: tuple[float, float, float, float]) -> tuple[float, float]`
  - `filter_to_pitch(points: list[tuple[float, float]], pitch_length: float = 105.0, pitch_width: float = 68.0, margin: float = 5.0) -> list[tuple[float, float]]`

  Used by Task 6 (`pipeline`) with these exact names and signatures.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_projection.py
import numpy as np
import pytest

from footsight.projection import (
    project_point,
    project_points,
    bbox_to_ground_point,
    filter_to_pitch,
)


def test_project_point_identity_homography():
    H = np.eye(3)
    assert project_point(H, (10.0, 20.0)) == (10.0, 20.0)


def test_project_point_scaling_homography():
    H = np.array([
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 1.0],
    ])
    result = project_point(H, (40.0, 60.0))
    assert result == pytest.approx((20.0, 30.0))


def test_project_points_applies_to_all():
    H = np.eye(3)
    points = [(1.0, 2.0), (3.0, 4.0)]
    assert project_points(H, points) == [(1.0, 2.0), (3.0, 4.0)]


def test_bbox_to_ground_point_uses_bottom_center():
    bbox = (10.0, 20.0, 30.0, 50.0)
    assert bbox_to_ground_point(bbox) == (20.0, 50.0)


def test_filter_to_pitch_keeps_points_within_margin():
    points = [(0.0, 0.0), (52.5, 34.0), (57.5, 39.0)]
    result = filter_to_pitch(points, pitch_length=105.0, pitch_width=68.0, margin=5.0)
    assert result == [(0.0, 0.0), (52.5, 34.0), (57.5, 39.0)]


def test_filter_to_pitch_discards_points_outside_margin():
    points = [(0.0, 0.0), (200.0, 0.0), (0.0, -100.0)]
    result = filter_to_pitch(points, pitch_length=105.0, pitch_width=68.0, margin=5.0)
    assert result == [(0.0, 0.0)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_projection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footsight.projection'`

- [ ] **Step 3: Write the implementation**

```python
# footsight/projection.py
import numpy as np


def project_point(H: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    """Apply a 3x3 homography to a single (x, y) pixel point."""
    x, y = point
    homogeneous = np.array([x, y, 1.0])
    result = H @ homogeneous
    result = result / result[2]
    return float(result[0]), float(result[1])


def project_points(H: np.ndarray, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [project_point(H, p) for p in points]


def bbox_to_ground_point(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Bottom-center of a bounding box (x1, y1, x2, y2) -- a player's ground-contact point."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def filter_to_pitch(
    points: list[tuple[float, float]],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    margin: float = 5.0,
) -> list[tuple[float, float]]:
    """Discard points that fall well outside the real pitch boundary (pitch centered at origin)."""
    half_length = pitch_length / 2.0 + margin
    half_width = pitch_width / 2.0 + margin
    return [
        (x, y) for x, y in points
        if -half_length <= x <= half_length and -half_width <= y <= half_width
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_projection.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add footsight/projection.py tests/test_projection.py
git commit -m "feat: add projection module (homography math, pitch filtering)"
```

---

### Task 3: `render` module

**Files:**
- Create: `footsight/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `pitch_to_image_coords(point: tuple[float, float], pitch_length: float = 105.0, pitch_width: float = 68.0, image_width_px: int = 1050, margin_px: int = 40) -> tuple[float, float]`
  - `render_pitch(player_positions: list[tuple[float, float]], output_path: str, pitch_length: float = 105.0, pitch_width: float = 68.0, image_width_px: int = 1050, margin_px: int = 40, dot_radius_px: int = 8) -> None`

  Used by Task 6 (`pipeline`) with these exact names and signatures.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_render.py
import pytest
from PIL import Image

from footsight.render import pitch_to_image_coords, render_pitch


def test_pitch_to_image_coords_center_of_pitch():
    px, py = pitch_to_image_coords((0.0, 0.0), image_width_px=1050, margin_px=40)
    assert px == pytest.approx(525.0)


def test_pitch_to_image_coords_corner_maps_to_margin():
    px, py = pitch_to_image_coords((-52.5, -34.0), image_width_px=1050, margin_px=40)
    assert px == pytest.approx(40.0)
    assert py == pytest.approx(40.0)


def test_render_pitch_creates_file_with_dot_at_expected_pixel(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch([(0.0, 0.0)], str(output_path), image_width_px=1050, margin_px=40, dot_radius_px=8)

    assert output_path.exists()
    image = Image.open(output_path)
    center_px, center_py = pitch_to_image_coords((0.0, 0.0), image_width_px=1050, margin_px=40)
    pixel = image.getpixel((int(center_px), int(center_py)))
    assert pixel == (220, 20, 60)


def test_render_pitch_with_no_players_still_creates_pitch(tmp_path):
    output_path = tmp_path / "empty.png"
    render_pitch([], str(output_path))
    assert output_path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footsight.render'`

- [ ] **Step 3: Write the implementation**

```python
# footsight/render.py
from PIL import Image, ImageDraw

PITCH_COLOR = (34, 139, 34)
LINE_COLOR = (255, 255, 255)
PLAYER_COLOR = (220, 20, 60)


def pitch_to_image_coords(
    point: tuple[float, float],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    image_width_px: int = 1050,
    margin_px: int = 40,
) -> tuple[float, float]:
    x, y = point
    scale = (image_width_px - 2 * margin_px) / pitch_length
    px = margin_px + (x + pitch_length / 2.0) * scale
    py = margin_px + (y + pitch_width / 2.0) * scale
    return px, py


def render_pitch(
    player_positions: list[tuple[float, float]],
    output_path: str,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    image_width_px: int = 1050,
    margin_px: int = 40,
    dot_radius_px: int = 8,
) -> None:
    scale = (image_width_px - 2 * margin_px) / pitch_length
    image_height_px = int(pitch_width * scale) + 2 * margin_px

    image = Image.new("RGB", (image_width_px, image_height_px), PITCH_COLOR)
    draw = ImageDraw.Draw(image)

    top_left = pitch_to_image_coords(
        (-pitch_length / 2, -pitch_width / 2), pitch_length, pitch_width, image_width_px, margin_px
    )
    bottom_right = pitch_to_image_coords(
        (pitch_length / 2, pitch_width / 2), pitch_length, pitch_width, image_width_px, margin_px
    )
    draw.rectangle([top_left, bottom_right], outline=LINE_COLOR, width=2)

    halfway_top = pitch_to_image_coords((0.0, -pitch_width / 2), pitch_length, pitch_width, image_width_px, margin_px)
    halfway_bottom = pitch_to_image_coords((0.0, pitch_width / 2), pitch_length, pitch_width, image_width_px, margin_px)
    draw.line([halfway_top, halfway_bottom], fill=LINE_COLOR, width=2)

    for position in player_positions:
        px, py = pitch_to_image_coords(position, pitch_length, pitch_width, image_width_px, margin_px)
        draw.ellipse(
            [px - dot_radius_px, py - dot_radius_px, px + dot_radius_px, py + dot_radius_px],
            fill=PLAYER_COLOR,
        )

    image.save(output_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_render.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add footsight/render.py tests/test_render.py
git commit -m "feat: add render module (placeholder pitch + player dots)"
```

---

### Task 4: `player_detection` module

**Files:**
- Create: `footsight/player_detection.py`
- Test: `tests/test_player_detection.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `PERSON_LABEL: int` (constant, value `1`)
  - `load_model() -> torchvision.models.detection.FasterRCNN`
  - `detect_players(image_path: str, model, confidence: float = 0.5) -> list[tuple[float, float, float, float]]`

  Used by Task 6 (`pipeline`) with these exact names and signatures.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_player_detection.py
import torch
from PIL import Image

from footsight.player_detection import detect_players, PERSON_LABEL


class FakeModel:
    def __init__(self, boxes, labels, scores):
        self._boxes = boxes
        self._labels = labels
        self._scores = scores

    def __call__(self, tensor):
        return [{
            "boxes": torch.tensor(self._boxes, dtype=torch.float32),
            "labels": torch.tensor(self._labels, dtype=torch.int64),
            "scores": torch.tensor(self._scores, dtype=torch.float32),
        }]


def _make_test_image(path):
    Image.new("RGB", (100, 100), color=(0, 128, 0)).save(path)


def test_detect_players_filters_to_person_label_above_confidence(tmp_path):
    image_path = tmp_path / "still.png"
    _make_test_image(image_path)

    model = FakeModel(
        boxes=[[10.0, 20.0, 30.0, 60.0], [5.0, 5.0, 15.0, 15.0], [40.0, 40.0, 50.0, 50.0]],
        labels=[PERSON_LABEL, 3, PERSON_LABEL],
        scores=[0.9, 0.95, 0.2],
    )

    boxes = detect_players(str(image_path), model, confidence=0.5)

    assert boxes == [(10.0, 20.0, 30.0, 60.0)]


def test_detect_players_returns_empty_list_when_no_people(tmp_path):
    image_path = tmp_path / "still.png"
    _make_test_image(image_path)

    model = FakeModel(boxes=[], labels=[], scores=[])

    boxes = detect_players(str(image_path), model)

    assert boxes == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_player_detection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footsight.player_detection'`

- [ ] **Step 3: Write the implementation**

```python
# footsight/player_detection.py
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor

PERSON_LABEL = 1  # COCO category index for "person" in torchvision's label convention (0 = background)


def load_model():
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return model


def detect_players(
    image_path: str,
    model,
    confidence: float = 0.5,
) -> list[tuple[float, float, float, float]]:
    image = Image.open(image_path).convert("RGB")
    tensor = to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(tensor)[0]

    boxes = []
    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        if label.item() == PERSON_LABEL and score.item() >= confidence:
            x1, y1, x2, y2 = box.tolist()
            boxes.append((x1, y1, x2, y2))
    return boxes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_player_detection.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add footsight/player_detection.py tests/test_player_detection.py
git commit -m "feat: add player_detection module (torchvision Faster R-CNN)"
```

---

### Task 5: `pitch_calibration` module (PnLCalib, isolated subprocess)

**Files:**
- Create: `vendor/PnLCalib` (git submodule)
- Create: `scripts/pnlcalib_infer.py`
- Create: `footsight/pitch_calibration.py`
- Create: `tests/fixtures/sample_still.png`
- Test: `tests/test_pitch_calibration.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `compute_homography(image_path: str, weights_kp: str = ..., weights_line: str = ..., device: str = "cpu") -> np.ndarray | None`

  Used by Task 6 (`pipeline`) with this exact name and signature. Returns `None` when calibration fails (see spec's error-handling section) — Task 6 must treat that as a hard failure, not a guess.

- [ ] **Step 1: Vendor PnLCalib as a git submodule**

```bash
git submodule add https://github.com/mguti97/PnLCalib.git vendor/PnLCalib
```

- [ ] **Step 2: Download the pretrained single-view weights**

```bash
curl -L -o weights/pnlcalib/SV_kp https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp
curl -L -o weights/pnlcalib/SV_lines https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines
```

- [ ] **Step 3: Copy one sample still into the test fixtures directory**

```bash
cp "stills/Screenshot 2026-09-10 at 11.58.59 AM.png" tests/fixtures/sample_still.png
```

(This copies, not moves — the original stays in `stills/` untouched.)

- [ ] **Step 4: Write the GPL-bound glue script**

This script directly imports PnLCalib internals, so — unlike everything else in this
plan — it is bound by PnLCalib's GPL-2.0 license if distributed. It runs only as a
subprocess (via `sys.executable`), never imported by `footsight` code.

```python
# scripts/pnlcalib_infer.py
"""
GPL-2.0: this script imports PnLCalib (https://github.com/mguti97/PnLCalib)
internals directly and is bound by PnLCalib's license. Run only as an
isolated subprocess from footsight — never import this module.
See docs/superpowers/specs/2026-09-10-core-cv-pipeline-design.md and
MEMORY.md, 2026-09-10, for why this boundary exists.
"""
import argparse
import json
import sys

import cv2
import numpy as np
import torch
import yaml

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib
from inference import inference, projection_from_cam_params


def load_models(weights_kp, weights_line, device):
    cfg = yaml.safe_load(open("config/hrnetv2_w48.yaml"))
    cfg_l = yaml.safe_load(open("config/hrnetv2_w48_l.yaml"))

    model = get_cls_net(cfg)
    model.load_state_dict(torch.load(weights_kp, map_location=device))
    model.to(device)
    model.eval()

    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(torch.load(weights_line, map_location=device))
    model_l.to(device)
    model_l.eval()

    return model, model_l


def compute_homography(image_bgr, model, model_l, device):
    h, w = image_bgr.shape[:2]
    cam = FramebyFrameCalib(iwidth=w, iheight=h, denormalize=True)
    final_params_dict = inference(
        cam, image_bgr, model, model_l,
        kp_threshold=0.3434, line_threshold=0.7867, pnl_refine=True,
    )
    if final_params_dict is None:
        return None

    P = projection_from_cam_params(final_params_dict)  # 3x4: world(meters, centered) -> image pixels
    H_world_to_image = P[:, [0, 1, 3]]                 # players stand on the Z=0 plane -> drop the Z column
    H_image_to_world = np.linalg.inv(H_world_to_image)
    return H_image_to_world.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights-kp", required=True)
    parser.add_argument("--weights-line", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(json.dumps({"error": f"could not read image: {args.image}"}), file=sys.stderr)
        sys.exit(1)

    model, model_l = load_models(args.weights_kp, args.weights_line, args.device)
    homography = compute_homography(image_bgr, model, model_l, args.device)
    print(json.dumps({"homography": homography}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write the failing test**

```python
# tests/test_pitch_calibration.py
from pathlib import Path

import numpy as np

from footsight.pitch_calibration import compute_homography

FIXTURE = Path(__file__).parent / "fixtures" / "sample_still.png"


def test_compute_homography_returns_invertible_3x3_or_none():
    # Integration test: runs the real PnLCalib subprocess against a real still.
    # Requires Task 5's setup steps (submodule + weights) to have been run.
    # Can take up to ~60s on CPU.
    homography = compute_homography(str(FIXTURE))

    if homography is None:
        return  # acceptable per the spec's error-handling section

    assert homography.shape == (3, 3)
    assert np.all(np.isfinite(homography))
    np.linalg.inv(homography)  # raises if degenerate/non-invertible
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_pitch_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footsight.pitch_calibration'`

- [ ] **Step 7: Write the implementation**

```python
# footsight/pitch_calibration.py
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PNLCALIB_VENDOR_DIR = REPO_ROOT / "vendor" / "PnLCalib"
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "pnlcalib_infer.py"
DEFAULT_WEIGHTS_KP = REPO_ROOT / "weights" / "pnlcalib" / "SV_kp"
DEFAULT_WEIGHTS_LINE = REPO_ROOT / "weights" / "pnlcalib" / "SV_lines"


def compute_homography(
    image_path: str,
    weights_kp: str = str(DEFAULT_WEIGHTS_KP),
    weights_line: str = str(DEFAULT_WEIGHTS_LINE),
    device: str = "cpu",
) -> np.ndarray | None:
    """Run PnLCalib in an isolated subprocess and return a 3x3 image->pitch homography.

    Runs out-of-process, never imported: PnLCalib is GPL-2.0 licensed. See
    scripts/pnlcalib_infer.py and MEMORY.md, 2026-09-10.
    """
    result = subprocess.run(
        [
            sys.executable, str(RUNNER_SCRIPT),
            "--image", str(Path(image_path).resolve()),
            "--weights-kp", str(Path(weights_kp).resolve()),
            "--weights-line", str(Path(weights_line).resolve()),
            "--device", device,
        ],
        cwd=PNLCALIB_VENDOR_DIR,
        env={**os.environ, "PYTHONPATH": str(PNLCALIB_VENDOR_DIR)},
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    if data["homography"] is None:
        return None
    return np.array(data["homography"])
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_pitch_calibration.py -v`
Expected: PASS (1 passed) — or a manual check that it exits cleanly if calibration
returns `None` on this particular still.

- [ ] **Step 9: Commit**

```bash
git add .gitmodules vendor/PnLCalib scripts/pnlcalib_infer.py footsight/pitch_calibration.py tests/test_pitch_calibration.py tests/fixtures/sample_still.png
git commit -m "feat: add pitch_calibration module (PnLCalib, isolated subprocess)"
```

---

### Task 6: `pipeline` orchestration + CLI

**Files:**
- Create: `footsight/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes:
  - `footsight.pitch_calibration.compute_homography(image_path, weights_kp, weights_line) -> np.ndarray | None` (Task 5)
  - `footsight.player_detection.load_model() -> model` and `detect_players(image_path, model, confidence=0.5) -> list[tuple]` (Task 4)
  - `footsight.projection.bbox_to_ground_point(bbox) -> tuple`, `project_points(H, points) -> list[tuple]`, `filter_to_pitch(points) -> list[tuple]` (Task 2)
  - `footsight.render.render_pitch(player_positions, output_path) -> None` (Task 3)
- Produces:
  - `run(input_path: str, output_path: str, detection_model, weights_kp: str, weights_line: str) -> None`
  - `main() -> None` (CLI entry point)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline.py
from unittest.mock import patch

import numpy as np
import pytest

from footsight import pipeline


@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.player_detection.detect_players")
@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_run_wires_stages_together(mock_compute_homography, mock_detect_players, mock_render_pitch):
    mock_compute_homography.return_value = np.eye(3)
    mock_detect_players.return_value = [(0.0, 0.0, 10.0, 20.0)]

    pipeline.run(
        "still.jpg", "mockup.png",
        detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
    )

    mock_compute_homography.assert_called_once_with("still.jpg", "kp.pt", "lines.pt")
    mock_detect_players.assert_called_once_with("still.jpg", "fake-model")
    mock_render_pitch.assert_called_once()
    rendered_points, output_path = mock_render_pitch.call_args[0]
    assert output_path == "mockup.png"
    assert rendered_points == [(5.0, 20.0)]


@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_run_raises_when_calibration_fails(mock_compute_homography):
    mock_compute_homography.return_value = None

    with pytest.raises(RuntimeError, match="Could not calibrate"):
        pipeline.run(
            "still.jpg", "mockup.png",
            detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'footsight.pipeline'`

- [ ] **Step 3: Write the implementation**

```python
# footsight/pipeline.py
import argparse

from footsight import pitch_calibration, player_detection, projection, render


def run(
    input_path: str,
    output_path: str,
    detection_model,
    weights_kp: str,
    weights_line: str,
) -> None:
    homography = pitch_calibration.compute_homography(input_path, weights_kp, weights_line)
    if homography is None:
        raise RuntimeError(f"Could not calibrate pitch from {input_path}")

    boxes = player_detection.detect_players(input_path, detection_model)
    ground_points = [projection.bbox_to_ground_point(box) for box in boxes]
    pitch_points = projection.project_points(homography, ground_points)
    pitch_points = projection.filter_to_pitch(pitch_points)

    render.render_pitch(pitch_points, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn a broadcast still into a 2D pitch mockup.")
    parser.add_argument("input_path", help="Path to the input still image")
    parser.add_argument("output_path", help="Path to save the rendered mockup PNG")
    parser.add_argument("--weights-kp", default=str(pitch_calibration.DEFAULT_WEIGHTS_KP))
    parser.add_argument("--weights-line", default=str(pitch_calibration.DEFAULT_WEIGHTS_LINE))
    args = parser.parse_args()

    detection_model = player_detection.load_model()
    run(args.input_path, args.output_path, detection_model, args.weights_kp, args.weights_line)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add footsight/pipeline.py tests/test_pipeline.py
git commit -m "feat: add pipeline orchestration and CLI entry point"
```

---

### Task 7: End-to-end manual validation + README

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: the full pipeline from Task 6.
- Produces: nothing new — this task is manual/visual validation, per the spec's testing section (calibration/detection accuracy isn't unit-testable).

- [ ] **Step 1: Run the full test suite**

```bash
pytest -v
```

Expected: all tests from Tasks 2-6 pass (Task 5's integration test may report calibration as `None` on a given still — that is a valid, spec'd outcome, not a failure).

- [ ] **Step 2: Run the pipeline against each provided sample still**

```bash
mkdir -p out
python -m footsight.pipeline "stills/Screenshot 2026-09-10 at 11.58.59 AM.png" out/mockup_1.png
python -m footsight.pipeline "stills/Screenshot 2026-09-10 at 11.59.40 AM.png" out/mockup_2.png
python -m footsight.pipeline "stills/Screenshot 2026-09-10 at 12.00.14 PM.png" out/mockup_3.png
```

- [ ] **Step 3: Visually compare each `out/mockup_N.png` against its source still**

Open both side by side. Check: are player dots roughly where the players actually
stand on the pitch? Is the pitch outline the right proportions? This is the
feasibility check this whole sub-project exists to answer — there is no
automated pass/fail here, per the spec.

- [ ] **Step 4: Write `README.md`**

```markdown
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
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup and usage instructions"
```

- [ ] **Step 6: Record the validation outcome in MEMORY.md**

Add an entry to `MEMORY.md` under today's date noting what the visual check in
Step 3 found (accurate enough / needs the classical-CV fallback for calibration /
other), since that finding determines whether this sub-project is "done" or
needs another pass — per the CLAUDE.md instruction to log outcomes of
significant checks.
