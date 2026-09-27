---
name: footsight
description: Working spec for footsight -- turns a football broadcast still into a 2D pitch mockup. Read before changing any footsight module: pipeline contract, module interfaces, hard-won lessons, and how to verify changes.
---

# footsight

Broadcast still (main wide camera) -> 2D top-down pitch mockup with players
colored by team, goalkeepers, referees and the ball. Personal-use v1, local
open-source tools only, zero per-image cost. Decisions and their history live
in `MEMORY.md`; failed approaches in `ERRORS.md`. This file is the distilled
contract -- keep it in sync when either changes.

## Pipeline (`footsight/pipeline.py::run`)

```
still.png
  -> pitch_calibration.compute_homography   3x3 image->pitch homography (or None -> RuntimeError)
  -> player_detection.detect_players        [(box, role)], role in player/goalkeeper/referee
  -> review(image_path, detections)         optional: --remove-detections drops false boxes
  -> projection (box bottom-center -> pitch meters), filter_to_pitch   off-pitch people dropped here
  -> team_classification.classify_players   only on-pitch role=="player" boxes -> team_a/team_b/officials
  -> team_classification.team_kit_colors    each team's (shirt, shorts) RGB, or {} -> fixed palette
  -> ball: ball_detection.find_ball, then optional ball_review (--pick-ball) to move/add/remove it
  -> render.render_pitch(..., kit_colors)   <out>.png  (top-down)
  -> pose.estimate_poses(still, on-pitch boxes, pose_model)   17x3 joints or None per person
  -> camera_view.render_camera_view(still, H, people, ball_pixel, <out>_camera.png)   camera angle
```

`people = [(box, category, (shirt, shorts), pose_or_None)]`, built once from
the on-pitch detections; both outputs share categories and kit colors
(`render.category_kit`). `run(pose_model=None)` draws standing figures only.

Category mapping: goalkeeper/referee render by detected role; a player the
classifier labels `officials` renders as `referee`.

## Module contracts

| Module | Interface | Rules |
|---|---|---|
| `pitch_calibration` | `compute_homography(image_path, weights_kp, weights_line, device="cpu") -> ndarray \| None` | PnLCalib (vendored submodule) runs **only as a subprocess** via `scripts/pnlcalib_infer.py`; never import it. |
| `player_detection` | `load_model(weights)`, `detect_players(image_path, model, confidence=0.7, image_size=1280) -> list[(box, role)]` | Ultralytics YOLO + Roboflow `football-player-detection-v9.pt`. Model's own `ball` class is dropped. |
| `detection_picker` | `detection_at(point, detections)`, `drop_detections(detections, removed)`, `review_detections(image_path, detections)` | Smallest box wins on overlap. Window loop has no automated test; checked by hand on 12.00.14 (2026-09-26). |
| `team_classification` | `classify_players(image_path, player_boxes) -> list[str]`, `team_kit_colors(image_path, player_boxes, labels) -> {team: (shirt, shorts)}` | See "Team split" below. `<2` players -> all `team_a`. Kit colors sample shirt at 25-45% and shorts at 50-62% of box height (grass masked); `{}` if a team is empty or the shirts are < 60 RGB apart. |
| `ball_detection` | `find_ball(image_path, player_boxes) -> box \| None` | Classical HSV + 5x5 opening + circularity >= 0.65, searched near players. The pipeline projects the ball from its box's bottom edge (ground contact), like a player. |
| `ball_picker` | `display_scale`, `to_source_pixel`, `handle_key(key, current, detected)`, `review_ball(image_path, detected)` | Enter accepts (incl. no ball), Esc keeps detection, Delete/Backspace removes. Clicks map back to full-resolution pixels. Window loop has no automated test; checked by hand on 12.00.14 (2026-09-26). |
| `projection` | `bbox_to_ground_point`, `project_points[_indexed]`, `filter_to_pitch` | Pitch coords in meters, **centered at origin**, 105 x 68, 5 m margin. |
| `pose` | `load_model(weights)`, `estimate_poses(image_path, boxes, model) -> [ndarray(17,3) \| None]`, plus `box_iou`, `crop_window`, `pick_person`, `is_reliable` | `yolo11m-pose.pt` on crops padded 35% sides / 20% top / 15% bottom, enlarged to 320 px, `conf=0.1`. `None` when < 10 joints > 0.3, IoU > 0.2 with another box, or skeleton height outside 0.6-1.4x the box. Missing weights -> `FileNotFoundError` naming the README step. |
| `posed_figure` | `draw_posed_player(image, pose, height_px, shirt, shorts)` | Cartoon human on the joints, sizes as fractions of H: head 0.075, torso min width 0.2 (side-on), thigh 0.10, calf 0.075; sleeves in shirt color, bare forearms, shorts over the upper half of the thigh, white socks, dark boots, outline, foot shadow; 4x supersampled. |
| `camera_view` | `render_camera_view(image_path, homography, people, ball_pixel, output_path, scale=1.5)`, `scaled_homography`, `vertical_scale`, `far_board_segments` | No broadcast pixels. Layers: seeded crowd dots, then top-down grass/markings warped by `inv(H)` (4.5 m apron, horizon-masked), then 0.9 m logo boards 4 m outside every side facing away from the camera (outward goes up the image), then goals in view (7.32 x 2.44 m frame, see-through net 2 m deep), then shadows/players/ball sorted far to near -- so keepers stand in front of their goal. Vertical px/m from a box-height vs foot-row fit (fallback 2.0x the ground px/m along the touchline). Ball radius max(4 px, 0.16 m). |
| `render` | `render_pitch(player_positions, output_path, ball_position=None, ..., image_width_px=4200, margin_px=160, player_height_m=1.8, ball_radius_m=0.33, kit_colors=None)` | `player_positions = [((x, y), category)]`; categories: team_a, team_b, goalkeeper, referee, assistant_referee. Pitch = 20 mown bands (5.25 m) with seeded grain + patch texture (identical output per input); markings 0.12 m wide, anti-aliased (3x mask). Icon = broadcast-style figure 15% above life size (1.8 m x 1.15) drawn 4x on its own tile, box-filtered down (flat areas keep exact colors), pasted so the figure's soles stand on the player's ground point (`ICON_FEET_UNITS`); the ball is drawn at its true position, so it only sits at a figure's feet when it did in the still. Detected kits are drawn with saturation x1.5 (`vivid_kit_color`); categories missing from `kit_colors` use the fixed shirt color + dark shorts. Procedural PIL, no assets. |

## Team split (the part most likely to break)

1. Torso sample = middle third of box height x middle half of width.
2. Drop grass pixels (OpenCV hue 40-60); keep all if nothing remains.
3. Per player compute **two** hues:
   - straight-line median -> Tukey fence (1.5 x IQR) trims odd colors as `officials` (a referee the detector called a player);
   - circular median (cut at the widest empty hue gap) -> k-means k=2 on unit-circle points for the team split.
4. Hue only. Saturation and value were tried and add noise (shadow, blur).

Why two hues: striped kits straddling the red seam (claret 170-179 and 0-10)
break a straight-line median; a circular trim lets the referee through because
striped teams spread widely on the circle.

## Lessons (don't relearn these)

- **Verify on real stills, not just tests.** Synthetic flat-color rectangles can't reproduce photo noise. Crop every doubtful box and look at it before concluding.
- **Watermarks** (Paramount+) score 0.65-0.73 as players -- overlapping real, partly hidden players (0.72). No confidence cutoff or box-shape filter separates them, and they project inside the pitch. Remove by hand.
- **Hue in the 40-60 band** means grass in the crop, not a jersey.
- **Ball boys** behind the touchline get detected as players (11.58.59, 12.00.14). They project about 5.5 m past the line and are dropped before the team split. The 5 m margin clears them by less than 1 m, so check where a box projects before calling it a player.
- Generic detectors (COCO) miss the ~12 px ball; a learned ball model flags spare balls by the touchline. Classical detection + manual fallback is the standing choice.
- Before changing a jersey feature, inspect per-player hue histograms -- that is what exposed the 50/50 striped-kit split.

## Brand assets

- Logo: `scripts/make_logo.py` -> `assets/logo/` (`footsight_logo.png` transparent, `footsight_logo_board.png` on navy, `footsight_ball.png`). "FOOTSiGHT", Kanit Black Italic (SIL OFL 1.1, read from `~/Library/Fonts`, not vendored), football at the font's own i-dot position, 1.875x the dot.
- Mockups must not reuse broadcast pixels (copyright, see MEMORY.md 2026-09-27): anything outside the pitch is drawn, not filtered from the still.

## Working conventions

- Use the project venv: `.venv/bin/python -m pytest -q`.
- Test-first: write the failing test, watch it fail for the right reason, then implement. Tests mock stages in `test_pipeline.py`; real-image checks go in a scratch script, not the suite.
- Sample stills are in `stills/`; macOS filenames contain a narrow no-break space before AM/PM -- use `glob`, not typed paths.
- Real-still reference results (as of 2026-09-26, full pipeline, watermark removed by hand): Bayern 5+7 + goalkeeper + ball; 11.58.59 10+10 + 1 referee + ball; 11.59.40 10+9 + 2 referees + ball; 12.00.14 10+10 + 2 referees, no ball.

## Open items

- A striped player whose straight-line median falls deep in a hue gap could still be trimmed as an official (not yet seen).
- No ball found on 12.00.14 (existing issue, not investigated; can now be added by hand with --pick-ball).
- Future direction: angled/perspective mockup camera with flat icons (needs its own design pass).
