# Camera-Angle View — Design Spec

Date: 2026-09-27
Status: Approved section by section in chat; pending written-spec review

## Context

footsight turns a broadcast still into a top-down 2D mockup. This adds a
second output drawn from the still's **own camera angle**: our striped pitch
and markings in perspective, a drawn stadium with footsight ad boards, and
cartoon players who copy each real player's pose. Every run writes both
images.

## Decisions carried in (see `MEMORY.md`, 2026-09-27 entries)

- **No broadcast pixels.** Everything outside the pitch is drawn, not
  filtered from the still. Filtering the frame keeps the broadcaster's image,
  and removing their watermark or scoreboard can itself be a problem. An
  image drawn only from positions and poses (facts) is on stronger ground.
  This is general understanding, not legal advice.
- No generative AI; open-source local models only (unchanged).
- **Poses** come from Ultralytics `yolo11m-pose.pt` (COCO 17 keypoints),
  run on enlarged crops of each player. A spike on 11.59.40 found a pose for
  all 21 players, mostly correct by eye.
- The ad boards carry `assets/logo/footsight_logo_board.png`.
- The stands are drawn as a crowd of dots (chosen from a sketch of three
  styles).
- The output is 1.5x the still's size. Measured: drawing takes 0.48 s at 1x,
  0.57 s at 1.5x and 0.67 s at 2x; posing takes about 1 s whatever the size.

## Scope

- Same input as today: one still, main wide camera. The same detection,
  detection review, team split, kit colors and ball review feed both outputs.
- Output: `<out>.png` (top-down, unchanged) plus `<out>_camera.png`.

## Architecture & components

```
pipeline.run(still, out_path, ...)
  1. calibrate (homography), detect, detection review        (unchanged)
  2. off-pitch filter, team split, kit colors, ball + review (unchanged)
  3. render.render_pitch(...)            -> <out>.png         (unchanged)
  4. pose.estimate_poses(still, boxes)   -> Pose | None per person   [new]
  5. camera_view.render_camera_view(...) -> <out>_camera.png         [new]
```

- **`footsight/pose.py`** (new)
  - `load_model(weights=DEFAULT_POSE_WEIGHTS)`, with default
    `weights/ultralytics/yolo11m-pose.pt`.
  - `estimate_poses(image_path, boxes, model) -> list[np.ndarray | None]`:
    one 17x3 array per box (x, y in still pixels, confidence), or `None`
    when the pose is unreliable.
- **`footsight/camera_view.py`** (new)
  - `render_camera_view(image_size, homography, people, ball_pixel,
    output_path, scale=1.5)`.
  - `people` is a list of `(box, category, (shirt, shorts), pose)`, where a
    `pose` of `None` means the standing figure is drawn.
- **`footsight/render.py`**: the helpers both renderers share become public,
  moved as-is with no behavior change: `grass_image`, `draw_markings`
  (anti-aliased), `draw_ball`, `draw_player_icon`.
- **`footsight/pipeline.py`**
  - New option `--pose-weights`.
  - Derives `<out>_camera.png` from the output path.
  - Builds `people` once and passes it to the camera renderer.

## Scene (camera_view)

**Scale.** All still-pixel inputs (boxes, poses, ball, homography) are
scaled by `scale` once, at the start.

**Layers, back to front:**

1. **Stands.** Everything above the boards' top edge gets a dark base plus
   crowd dots. The dots grow slightly toward the bottom, come from a general
   color palette, and use a fixed seed, so the output is the same on every
   render.
2. **Ground.** The top-down grass and markings are warped into the camera's
   view with `inv(H)`.
   - The grass extends 4 m beyond the touchlines and goal lines, up to the
     boards.
   - Anything beyond the horizon is excluded.
3. **Ad boards.**
   - They stand 4 m outside the lines, 0.9 m tall, drawn only along the
     sides facing away from the camera.
   - **Height on screen:** the homography only covers the ground plane, so
     the vertical pixels per metre at any image row come from a linear fit
     of detected player box height against foot row, taking a player as
     1.8 m. With fewer than 2 players, vertical px/m is taken as 2.0x the
     ground px/m along the goal-line direction at that row. That constant is
     rough and will be checked against the fit on the 4 sample stills.
   - Navy board, with the logo repeated about every 10 m. Each copy is
     warped to its section's four corners.
   - Sections beyond the horizon are skipped.
4. **Shadows, players and ball.** All are sorted far to near by their ground
   point, so players are always in front of the boards and a nearer player
   overlaps a farther one. The ball has a minimum on-screen radius.

## Posed players

**Crop.** Each box is padded 35% on each side, 20% above and 15% below, then
enlarged to 320 px tall. The model runs one crop at a time with `conf=0.1`
and `imgsz=320`. Among the people found in a crop, the one whose box overlaps
the target box most (by IoU) is used.

**Unreliable pose (`None`)** when any of these holds:
- fewer than 10 of the 17 joints have confidence above 0.3;
- the box overlaps another person's box with IoU above 0.2 (both people fall
  back);
- the skeleton's height (head to ankles) is outside 0.6-1.4x the box height.

**Drawing.** H is the figure's height from its box. The figure is drawn 4x
larger and box-filtered down, with a thin dark outline. Draw order: legs,
shorts, torso, arms, head.

| Part | Drawing |
|---|---|
| Head | circle, radius 0.075 H, centered on the face joints, with a hair cap |
| Torso | rounded shape from shoulders to hips, in the shirt color; minimum width about 0.2 H (side-on players); slightly fuller at chest and hips |
| Arms | upper arm as a shirt-color sleeve; forearm and hand in skin tone |
| Legs | tapered thigh (0.1 H) and calf (0.075 H); shorts over the upper half of the thigh; white socks on the lower calf; dark boot at the ankle |
| Shadow | soft ellipse at the feet |

Goalkeepers and referees are posed the same way, in their fixed colors. The
fallback is today's standing figure, sized from the box and placed at its
bottom-center.

## Error handling

- **Pose weights missing:** stop before any work, with a message pointing to
  the README download step.
- **Calibration fails:** raises, as today.
- **No players:** the stadium and pitch only. The board height falls back to
  the ground scale.
- **Horizon:** ground and board sections beyond the horizon are not drawn.

## Setup

Add a README step:

```
mkdir -p weights/ultralytics
curl -L -o weights/ultralytics/yolo11m-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt
```

The file is 42.5 MB (measured). Ultralytics is AGPL, which is fine for the
personal-use v1 (MEMORY.md, 2026-09-22).

## Testing (test-first)

- **pose:** crop geometry; mapping joints back to still pixels; picking a
  person by IoU; each reliability rule. Uses a fake model, following the
  `FakeModel` pattern in `tests/test_player_detection.py`.
- **camera_view:**
  - output size is 1.5x the input;
  - the vertical-scale fit, and its fallback with fewer than 2 players;
  - board quads against a known test homography;
  - a player is drawn in front of a board;
  - the crowd is identical on repeat renders;
  - a posed figure has the shirt color between shoulders and hips and skin
    on the forearms;
  - a `None` pose draws the standing figure.
- **pipeline:** both files are written; the pose step only receives on-pitch
  boxes; both renderers get the same categories and colors (mocks).
- **Real stills:** all 4 rendered and checked by eye.

**Expected speed:** about +2 s per run (pose about 1 s, drawing about 0.6 s,
plus saving).

## Out of scope

- Animation or video.
- Crowd tinted in team colors.
- Boards on the camera's side.
- Shirt numbers and facing direction.
- Fitting a 3D body model.
