# Camera-Angle View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every run also writes `<out>_camera.png`: the still's own camera angle, redrawn with our pitch, a drawn stadium with footsight boards, and cartoon players posed like the real ones.

**Architecture:**
- New `pose.py`: Ultralytics pose model on enlarged crops, with reliability rules.
- New `posed_figure.py`: draws a cartoon human from 17 joints.
- New `camera_view.py`: geometry and the layered scene.
- `render.py`: exposes its shared helpers publicly, with no behavior change.
- `pipeline.py`: builds the list of people once and writes both images.

**Tech Stack:** Python 3.14, Pillow, NumPy, OpenCV, Ultralytics YOLO (`yolo11m-pose.pt`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-camera-view-design.md`

## Global Constraints

- No broadcast pixels in the camera view. Everything outside the pitch is drawn.
- No generative AI; local open-source models only.
- Output scale: 1.5x the still.
- Pose model: `weights/ultralytics/yolo11m-pose.pt`, loaded with `conf=0.1` and `imgsz=320`. Crops are enlarged to 320 px tall and padded 35% on each side, 20% above and 15% below.
- A pose is unreliable when:
  - fewer than 10 of 17 joints are confident (> 0.3);
  - the box overlaps another box with IoU > 0.2;
  - the skeleton height is outside 0.6-1.4x the box height.
- Boards stand 4 m outside the lines, are 0.9 m tall, and appear only on the sides facing away from the camera. The logo repeats every 10 m.
- Board height on screen comes from a linear fit of box height against foot row, taking a player as 1.8 m. With fewer than 2 players, use 2.0x the ground px/m along the touchline direction.
- The crowd uses a fixed seed and a dark base, with dots growing toward the bottom.
- Draw order: stands, ground, boards, then shadows, players and ball sorted far to near.
- Test-first for every new behavior. Commit once at the end (user instruction for this build).
- Run tests with `.venv/bin/python -m pytest -q`.

---

### Task 1: Setup and shared render helpers

**Files:**
- Modify: `footsight/render.py`
  - rename `_grass` -> `grass_image`, `_draw_smooth_markings` -> `draw_markings`, `_draw_ball` -> `draw_ball`, `_draw_player_icon` -> `draw_player_icon`;
  - add `category_kit`.
- Create: `weights/ultralytics/yolo11m-pose.pt`, copied from the scratchpad spike download (same file, 42,459,307 bytes). This is not a new download.
- Modify: `README.md` (setup step for the pose model).
- Test: `tests/test_render.py`

**Interfaces:**
- Produces:
  - `render.grass_image(w, h, margin_px, scale) -> Image`
  - `render.draw_markings(image, pitch_length, pitch_width, margin_px, scale) -> None`
  - `render.draw_ball(image, position_px, radius_px) -> None`
  - `render.draw_player_icon(image, feet_px, shirt, shorts, unit_px) -> None`
  - `render.category_kit(category, kit_colors) -> (shirt, shorts)`: returns boosted detected colors when the category is in `kit_colors`, otherwise the fixed shirt color with `DEFAULT_SHORTS_COLOR`.

- [ ] Write a failing test that `category_kit("team_a", {"team_a": ((156, 62, 45), (240, 240, 240))})` returns `(vivid_kit_color((156, 62, 45)), (240, 240, 240))`, and that `category_kit("goalkeeper", {})` returns `(GOALKEEPER_COLOR, DEFAULT_SHORTS_COLOR)`.
- [ ] Run it and confirm it fails (ImportError).
- [ ] Rename the four helpers, update the call sites in `render_pitch`, add `category_kit`, and use it in `render_pitch`.
- [ ] Run the full suite: everything green, with no output change for the top-down mockup.
- [ ] Copy the pose weights and add the README step:
  ```
  mkdir -p weights/ultralytics
  curl -L -o weights/ultralytics/yolo11m-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt
  ```

### Task 2: `pose.py`

**Files:**
- Create: `footsight/pose.py`
- Test: `tests/test_pose.py`

**Interfaces:**
- Produces:
  - `pose.DEFAULT_POSE_WEIGHTS`
  - `pose.load_model(weights)`: raises `FileNotFoundError` with a README hint when the file is missing.
  - `pose.box_iou(a, b) -> float`
  - `pose.crop_window(box, image_w, image_h) -> (x1, y1, x2, y2)` as ints
  - `pose.pick_person(person_boxes, target_box) -> int | None`: index with the highest IoU; `None` if the list is empty or the best IoU is 0.
  - `pose.is_reliable(pose, box, other_boxes) -> bool`
  - `pose.estimate_poses(image_path, boxes, model) -> list[np.ndarray | None]`: each pose is a (17, 3) array of x, y in still pixels plus confidence.

Tests use a fake model: its `predict(image, conf, imgsz, verbose)` returns one object with `.boxes.xyxy` and `.keypoints.xy` / `.keypoints.conf` (NumPy-backed, `.cpu().numpy()`-compatible).

- [ ] Failing tests:
  - `box_iou` of identical boxes is 1, of disjoint boxes 0, of half-overlap 1/3;
  - `crop_window((100, 100, 120, 160), 1000, 1000)` equals `(93, 88, 127, 169)`, and it clamps at the image edges;
  - `pick_person` chooses the best overlap and returns `None` for an empty list;
  - `is_reliable` is False for 9 confident joints, False for IoU 0.3 with another box, False for a skeleton 0.5x the box height, and True for a clean pose;
  - `estimate_poses` maps crop keypoints back to still pixels (fake keypoints at known crop coordinates), and returns `None` when the model finds no one.
- [ ] Run them and confirm they fail.
- [ ] Implement.
- [ ] Run and confirm they pass.

### Task 3: `posed_figure.py`

**Files:**
- Create: `footsight/posed_figure.py`
- Test: `tests/test_posed_figure.py`

**Interfaces:**
- Consumes: a pose (17, 3) in image pixels, plus `height_px` (box height).
- Produces: `posed_figure.draw_posed_player(image, pose, height_px, shirt, shorts) -> None`
  - Parts: head radius 0.075 H; torso minimum width 0.2 H; thigh 0.1 H; calf 0.075 H; sleeves in the shirt color; forearms in skin; white socks; dark boots; outline; foot shadow.
  - Drawn 4x supersampled on its own tile.
  - COCO keypoint order: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists, 11-12 hips, 13-14 knees, 15-16 ankles.

- [ ] Failing tests on a synthetic standing pose (built in the test from a 200 px tall skeleton):
  - the pixel midway between the shoulder midpoint and the hip midpoint is the shirt color;
  - the forearm midpoint is the skin color;
  - a side-on pose (both shoulders at the same x) still has the shirt color at least 0.08 H either side of the torso center line.
- [ ] Run them and confirm they fail.
- [ ] Implement.
- [ ] Run and confirm they pass.

### Task 4: `camera_view.py`

**Files:**
- Create: `footsight/camera_view.py`
- Test: `tests/test_camera_view.py`

**Interfaces:**
- Consumes: the Task 1 render helpers, `pose.is_reliable` output (`None` poses), and `posed_figure.draw_posed_player`.
- Produces:
  - `camera_view.scaled_homography(H, scale) -> ndarray`: image->pitch for the scaled image, `H @ inv(diag(s, s, 1))`.
  - `camera_view.vertical_scale(boxes, H_scaled) -> Callable[[float], float]`: px per vertical metre at an image row, from the linear fit, with the fallback.
  - `camera_view.far_board_segments(H_scaled, image_w, image_h) -> list[(base_left, base_right)]`: ground points of 10 m board sections, on the far sides only, in front of the camera.
  - `camera_view.render_camera_view(image_path, homography, people, ball_pixel, output_path, scale=1.5) -> None`
    - `people = [(box, category, (shirt, shorts), pose_or_None)]`, in still pixels.

- [ ] Failing tests:
  - `vertical_scale` from boxes of height 90 at row 900 and 60 at row 600 gives 50 px/m at row 900 (90 / 1.8) and ~41.7 at row 750;
  - with a single box, it uses the fallback, 2.0x the ground px/m along the touchline;
  - `far_board_segments` with a synthetic homography (camera looking up the pitch, far side at the top of the image) returns segments on the y = -38 m line only;
  - the output image is 1.5x the still's size;
  - the crowd is identical on two renders;
  - a player whose box overlaps the board strip is drawn in front of the board (shirt color present at the player's torso);
  - a `None` pose draws the standing icon (shirt color present);
  - a posed person gets `draw_posed_player` (patched spy called once).
  - These tests use a small synthetic still and a synthetic homography built from 4 point correspondences with `cv2.getPerspectiveTransform`.
- [ ] Run them and confirm they fail.
- [ ] Implement: stands, then the ground warp (horizon-masked), boards with the warped logo tiles, then sorted drawables (players plus ball, with ball radius `max(4, 0.3 m * vertical_scale(row))`).
- [ ] Run and confirm they pass.

### Task 5: Pipeline wiring

**Files:**
- Modify: `footsight/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `pose.load_model`, `pose.estimate_poses`, `camera_view.render_camera_view`, `render.category_kit`.
- Produces:
  - `pipeline.camera_output_path(output_path) -> str`: `a/b.png` becomes `a/b_camera.png`.
  - `run(..., pose_model=None)`: with `None`, every pose is `None` (standing figures).
  - `main()`: `--pose-weights`.

- [ ] Failing tests:
  - `camera_output_path`;
  - `run` calls `render_camera_view` once with `camera_output_path(out)`, the on-pitch people with categories matching the top-down mockup, and the reviewed ball pixel;
  - `estimate_poses` receives only on-pitch boxes when a `pose_model` is given;
  - an autouse fixture stubs `render_camera_view` and `estimate_poses` for all pipeline tests.
- [ ] Run them and confirm they fail.
- [ ] Implement.
- [ ] Run and confirm they pass.

### Task 6: Real-still check, docs, commit

- [ ] Run the pipeline on all 4 stills, with the watermark removed on the Barça stills. Check both outputs by eye; fix any issues test-first.
- [ ] Update `SKILL.md` (modules, pipeline diagram), `MEMORY.md` (build entry), `ERRORS.md` (if anything took more than 2 attempts), and fix the touchline wording in the spec.
- [ ] Run the full suite.
- [ ] Make one commit with all changes (user-approved in advance for this build).
