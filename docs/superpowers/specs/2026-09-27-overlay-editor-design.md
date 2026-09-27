# Overlay Editor (Live Telestrator) — Design Spec

Date: 2026-09-27
Status: Approved and implemented (plan: docs/superpowers/plans/2026-09-27-overlay-editor.md)

## Context

footsight's audience makes football content: they grab stills from a match
and narrate over them. This adds a browser editor for drawing analysis
graphics on the **camera-angle view** while narrating and screen-recording,
like a TV telestrator. It adds player highlights and tags, run and pass
arrows, links, lines and zones, all drawn live.

## Decisions (chosen in chat, 2026-09-27)

- **Tools:** highlight & tag players; arrows and lines; zones. Measurements
  are not in v1.
- **Views:** camera view only. The top-down mockup has no overlays.
- **Editor:** a local web app in the browser, served from the user's machine.
  Nothing goes online.
- **Style:** broadcast. Bright colors with a soft glow; dark name tags with a
  colored border (picked from a sketch over "flat").
- **Workflow:** live while recording, so:
  - full-screen;
  - one key per tool;
  - graphics appear as you draw;
  - quick undo and clear;
  - step through a folder of stills.
- **Saving is optional and explicit:** Export PNG and/or Save overlays
  (reopenable). Nothing is written otherwise; closing discards the drawings.
- **Graphics are stored in pitch metres** and tied to player IDs, so
  ground graphics lie on the grass in perspective.

## Architecture

Approach: a layered camera view plus a browser editor. Ground graphics sit
between the background and figures layers, so they're under the players,
and what's on screen is exactly what's exported.

Rejected alternatives:
- Browser sends the overlays and Python redraws the final image: two
  drawing engines that could drift apart.
- Overlays on top of the flat image: the graphics would cover the players.

```
pipeline.run(...)  -> out_camera.png                  (unchanged)
                   -> out_camera_background.png  [new] stadium, pitch, boards, goals
                   -> out_camera_figures.png     [new] shadows, players, ball (transparent)
                   -> out_camera_scene.json      [new] players + camera geometry

python -m footsight.edit <folder-or-still> [--port 8765] [--no-browser]   [new]
```

- **`footsight/camera_view.py`**
  - Renders the background and figures as separate images. Figures are
    drawn onto a transparent layer; the full image is background + figures.
  - Also saves them as `_background.png` and `_figures.png`.
  - Writes `_scene.json`.
- **`footsight/pipeline.py`**: no interface change. The paths are derived
  from `camera_output_path`.
- **`footsight/edit.py`** (new): a standard-library HTTP server
  (`http.server`), bound to 127.0.0.1, which opens the browser.
- **`footsight/editor/`** (new; static, no build step, no npm packages):
  - `index.html`, `editor.css`;
  - `editor.js`: UI, tools, keys, SVG, export;
  - `geometry.js`: pure functions, unit-tested with `node --test`;
  - `tests/geometry.test.mjs`.

## Scene file (`<still>_camera_scene.json`)

```json
{
  "version": 1,
  "image": {"width": 4536, "height": 2553,
            "background": "out_camera_background.png", "figures": "out_camera_figures.png"},
  "pitch": {"length": 105, "width": 68},
  "image_to_pitch": [[...], [...], [...]],
  "pitch_to_image": [[...], [...], [...]],
  "players": [
    {"id": 0, "category": "team_a", "shirt": [133,190,228], "shorts": [201,224,244],
     "feet": [1512.0, 1480.5], "feet_m": [-18.2, 6.4], "box": [1480, 1330, 1545, 1481]}
  ],
  "ball": {"pixel": [2090.0, 1402.0], "m": [-4.1, 3.9]}
}
```

- Pixels are camera-view pixels (the 1.5x image); `_m` values are pitch
  metres (origin at the center spot).
- `pitch_to_image` is the inverse of `image_to_pitch`.
- Player IDs are their order in `players`. `ball` is `null` when there is no
  ball.

## Tools (live-first)

| Key | Tool | Behavior | Default color |
|---|---|---|---|
| H | Highlight | Click a player: a glowing 1.3 m ring on the grass around the feet. Click again to remove it. | yellow |
| T | Tag | Click a player, type, Enter: the tag floats above the head (box top) at a fixed on-screen size. | yellow border |
| A | Run arrow | Drag: a solid 0.55 m ribbon on the grass along the smoothed drag path, with a 2 m head. Starting on a player snaps the start to their feet. | white |
| P | Pass arrow | Same as a run arrow, dashed, 0.4 m wide. | yellow |
| K | Link | Click players in turn: a line joining their feet. Enter or double-click finishes; Esc cancels. | cyan |
| L | Line | Drag: a straight line on the grass. **Shift:** across the pitch, touchline to touchline, at the cursor's pitch-length position. **Alt:** dashed. | cyan |
| Z | Zone | Click points on the grass; Enter or double-click closes into a translucent fill with a border. | yellow |
| S | Spotlight | Toggle: dims everything except the highlighted players. | — |

**Other keys:**
- 1-4: yellow / cyan / white / red for the next graphic.
- Cmd/Ctrl+Z undo; Shift+Cmd/Ctrl+Z redo.
- C clears the still; Esc cancels the shape in progress.
- ←/→ previous/next still. Drawings are kept per still for the whole
  session.
- F full screen; the toolbar hides until the mouse reaches the top edge.
- E export PNG; Cmd/Ctrl+S save overlays.

**Colors (broadcast):**
- yellow `#FFD60A`, cyan `#00DCFF`, white `#FFFFFF`, red `#FF3B30`;
- ground fills at about 27% opacity;
- glow: an SVG Gaussian blur merged under each shape;
- tags: dark `#0F1423` at 90% with a colored border, Kanit SemiBold.
  - The server exposes `GET /editor/fonts/kanit-semibold.ttf`, streamed from
    `~/Library/Fonts/Kanit-SemiBold.ttf` (SIL OFL; not copied into the repo,
    matching the logo script).
  - If the font isn't there, tags fall back to the system bold sans-serif.

**Layer order:**
1. background;
2. ground graphics (rings, arrows, links, lines, zones);
3. figures;
4. spotlight veil;
5. tags.

**Out of v1:** select-and-move, measurements, animation playback, top-down
overlays, video.

## Geometry (`geometry.js`, pure)

- `applyHomography(matrix, [x, y]) -> [x, y] | null`: `null` when the point
  is beyond the horizon (the w sign differs from the pitch center's).
- `toPitch(scene, px)` and `toImage(scene, m)`.
- `snapPlayer(scene, px) -> id | null`: a player whose box contains the
  point, otherwise the nearest one whose feet are within 3 m on the pitch.
- `ringPolygon(center_m, radius_m=1.3, n=48)`.
- `ribbonPolygon(path_m, width_m)`.
- `arrowHead(tip_m, direction, size_m=2)`.
- `densify(path_m, step_m=0.25)` before projecting, so curves stay smooth
  in perspective.
- `smoothPath(points_m)`: Ramer-Douglas-Peucker at 0.5 m, then Catmull-Rom.
- `acrossPitchLine(x_m, pitch) -> [[x, -34], [x, 34]]`.
- `serializeOverlays` / `parseOverlays`: format `{"version": 1, "overlays":
  [...], "spotlight": bool}`, with overlays of type:
  - `ring {player, color}`
  - `tag {player, text, color}`
  - `arrow {style: run|pass, points_m, color}`
  - `link {players, color}`
  - `line {from_m, to_m, dashed, color}`
  - `zone {points_m, color}`

The SVG viewBox is the image size; the pointer is converted with
`getScreenCTM`, so drawing works at any window size.

## Server (`footsight/edit.py`)

- **Stills:** found from `*_camera_scene.json` in the folder (or the single
  still given), sorted by name.
- **Bind:** 127.0.0.1 only. If the port is taken, a free one is chosen.
  `webbrowser.open` unless `--no-browser`.
- **Endpoints:**
  - `GET /` and `GET /editor/<file>`: editor files only, plus the tag font
  (see Colors).
  - `GET /api/stills`: `[{id, name, has_overlays}]`.
  - `GET /api/stills/<id>/scene | background | figures | overlays`: 404 when
    there are no saved overlays.
  - `POST /api/stills/<id>/overlays`: JSON, validated by `version` and list
    shape; writes `<still>_camera_overlays.json`.
  - `POST /api/stills/<id>/export`: PNG bytes (signature checked, ≤ 50 MB);
    writes `<still>_camera_edited.png`.
- Only files of discovered stills are served; there's no path parameter
  mapped to disk.

## Errors

- **No stills found:** exit with "run the pipeline on your stills first".
- **Still missing its layer or scene files** (generated before this
  change): skipped, and listed in the editor with a "re-run the pipeline"
  note.
- **Save or export failure:** an on-screen notice; the drawings stay in
  the browser.

## Testing (test-first)

- **pytest (camera_view):**
  - no player color at a player's torso in the background layer;
  - the figures layer is transparent away from figures;
  - background + figures equals `out_camera.png`;
  - scene file players match their `feet` and `feet_m` through the matrices;
  - the two matrices are inverses;
  - `ball` is null without a ball.
- **pytest (edit):** a server on an ephemeral port in a thread:
  - list, scene, layers, 404 on no overlays;
  - save overlays round-trip;
  - export rejects non-PNG and oversize uploads;
  - unknown still IDs give 404;
  - the server address is 127.0.0.1.
- **node --test (geometry.js):**
  - projection round-trip and the horizon;
  - snapping, both inside a box and by distance;
  - ring and ribbon shapes;
  - densify and smoothPath;
  - acrossPitchLine;
  - the overlay format round-trip.
- **Manual live checklist for the user:**
  - each tool and key;
  - full screen;
  - stepping through stills;
  - undo and clear;
  - export and save/reopen.
