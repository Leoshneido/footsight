# Overlay Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local browser telestrator: `python -m footsight.edit <folder>` opens the camera views full-screen. Broadcast-style graphics (highlight, tag, run/pass arrows, link, line, zone, spotlight) are drawn live on the grass under the players, with optional export and save.

**Architecture:**
- `camera_view` also writes a background layer, a transparent figures layer and a scene file.
- A standard-library server (`edit.py`) serves them, plus a static editor (`footsight/editor/`: HTML, CSS, `editor.js`, `geometry.js`).
- Graphics are built in pitch metres and projected with the scene's homography into SVG, stacked between the two image layers.

**Tech Stack:** Python 3.14 (`http.server`, Pillow, NumPy), plain JavaScript (ES modules) and SVG, pytest, `node --test` (Node 25).

**Spec:** `docs/superpowers/specs/2026-09-27-overlay-editor-design.md`

## Global Constraints

- Camera view only; the top-down mockup is untouched.
- The server binds 127.0.0.1 only and serves only files of discovered stills and editor files. Exports must be PNG, ≤ 50 MB.
- Nothing is written unless the user exports (E) or saves (Cmd/Ctrl+S).
- Broadcast colors: yellow `#FFD60A`, cyan `#00DCFF`, white `#FFFFFF`, red `#FF3B30`. Fills at ~27% opacity; SVG glow; tags dark `#0F1423` at 90% with a colored border, Kanit SemiBold (streamed from `~/Library/Fonts`, falling back to system bold sans).
- Sizes: ring 1.3 m; run ribbon 0.55 m; pass 0.4 m dashed; arrowhead 2 m; densify every 0.25 m; smoothing RDP 0.5 m then Catmull-Rom; snapping inside the box or within 3 m of the feet.
- Keys:
  - tools: H, T, A, P, K, L (Shift = across the pitch, Alt = dashed), Z, S;
  - colors: 1-4;
  - undo/redo: Cmd/Ctrl+Z and Shift+Cmd/Ctrl+Z;
  - C clear, Esc cancel, ←/→ stills, F full screen, E export, Cmd/Ctrl+S save.
- Layer order: background, ground graphics, figures, spotlight veil, tags.
- Test-first; one commit at the end (user instruction for this build).
- Tests: `.venv/bin/python -m pytest -q` and `node --test footsight/editor/tests/*.test.mjs` (a folder argument fails).

---

### Task 1: Camera-view layers and scene file

**Files:** Modify `footsight/render.py`, `footsight/posed_figure.py`, `footsight/camera_view.py`. Test `tests/test_camera_view.py`.

**Interfaces:**
- `render.paste_over(target, tile, xy)`: onto RGB it pastes with the tile's alpha as mask (today's behavior); onto RGBA it does a clipped `alpha_composite`, so transparent layers keep correct edges.
- `draw_player_icon`, `draw_ball` and `draw_posed_player` paste through `paste_over`.
- `camera_view.layer_paths(output_path) -> {"background", "figures", "scene"}`: `x/out_camera.png` gives `x/out_camera_background.png`, `x/out_camera_figures.png` and `x/out_camera_scene.json`.
- `render_camera_view` writes all four files. The full image is `background.alpha_composite(figures)`.

- [ ] Failing tests:
  - `layer_paths`;
  - the background has no shirt color at a player's torso, while the full image does;
  - the figures layer is transparent (alpha 0) at a spot on the grass away from any figure;
  - background composited with figures equals the full image;
  - scene: `players[i].feet` maps through `image_to_pitch` to `feet_m`, and `feet_m` maps back through `pitch_to_image`;
  - the matrices multiply to the identity (up to scale);
  - IDs 0..n-1 in order; category and colors copied;
  - `ball` is null without a ball and has `pixel` and `m` with one.
- [ ] Implement; run the full suite green (the top-down mockup is unchanged).

### Task 2: `geometry.js`

**Files:** Create `footsight/editor/geometry.js` and `footsight/editor/tests/geometry.test.mjs`.

**Interfaces** (ES module exports):
- `applyHomography(m, [x, y])`: returns `[x, y]`, or `null` beyond the horizon when given a reference sign.
- `makeProjector(scene)`: returns `{ toPitch(px), toImage(m) }`, where `toImage` returns `null` beyond the horizon.
- `snapPlayer(scene, projector, px, maxMeters=3)`
- `ringPolygon(c, r=1.3, n=48)`
- `densify(path, step=0.25)`
- `ribbonPolygon(path, width)`
- `arrowHead(tip, dir, size=2)`
- `simplifyRDP(points, eps=0.5)`
- `catmullRom(points, samples=8)`
- `smoothPath(points)`: RDP, then Catmull-Rom.
- `acrossPitchLine(x, pitch)`
- `projectPolygon(projector, points)`: drops points beyond the horizon.
- `serializeOverlays(overlays, spotlight)` and `parseOverlays(json)`: `parseOverlays` throws on a bad version or shape.

- [ ] Failing node tests for each function: round-trips, a horizon `null`, snapping inside a box and by distance, ring radius, ribbon width, densify spacing, RDP removing collinear points, smoothPath endpoints kept, the across-pitch line spanning −34..34, the overlay round-trip, and rejection of bad input.
- [ ] Implement; `node --test` green.

### Task 3: `edit.py` server

**Files:** Create `footsight/edit.py`. Test `tests/test_edit.py`.

**Interfaces:**
- `find_stills(path) -> list[Still]`: `Still` has name and the paths to the scene, background, figures, full image, overlays and edited PNG. Stills missing a layer go in a separate `skipped` list: `find_stills` returns `(stills, skipped)`.
- `make_server(stills, skipped, port=0) -> ThreadingHTTPServer`: bound to 127.0.0.1.
- `main()`: argparse `path`, `--port 8765` (falls back to a free port), `--no-browser`.
- Routes:
  - `GET /` gives `editor/index.html`;
  - `GET /editor/<name>` gives files in `footsight/editor/` only (no `..`), plus `fonts/kanit-semibold.ttf`;
  - `GET /api/stills` gives `{stills: [{id, name, has_overlays}], skipped: [names]}`;
  - `GET /api/stills/<id>/scene|background|figures|overlays`;
  - `POST /api/stills/<id>/overlays` (validated JSON);
  - `POST /api/stills/<id>/export` (PNG signature, ≤ 50 MB).

- [ ] Failing tests with a tmp folder holding one generated still (tiny layer PNGs and a scene) plus one missing a layer:
  - the list and the skipped list;
  - serving the scene and layers;
  - 404 when there are no overlays;
  - the overlay save round-trip plus `has_overlays`;
  - 400 on invalid overlays;
  - export writes the PNG; 400 on non-PNG; 413 when oversize;
  - 404 on an unknown ID and on `/editor/../edit.py`;
  - the server address is 127.0.0.1.
- [ ] Implement; green.

### Task 4: Editor UI

**Files:** Create `footsight/editor/index.html`, `editor.css` and `editor.js`.

- [ ] Build the UI in `editor.js`:
  - an SVG stage at the image size: background `<image>`, ground `<g>` with a glow filter, figures `<image>`, spotlight mask, tags `<g>`;
  - the tool state machine for all keys, with a live preview while drawing;
  - undo/redo stacks per still;
  - ←/→ navigation keeping drawings per still;
  - full-screen and an auto-hiding toolbar that shows the active tool and color;
  - export: rasterize the SVG to a canvas at the image size, then POST;
  - save: POST the serialized overlays; load saved overlays on open;
  - on-screen notices;
  - the skipped-still notice.
- [ ] Automated checks:
  - `node --check` on `editor.js`;
  - a pytest that the server serves `index.html` and `editor.js`, and that `index.html` references the editor module and the stage.
- [ ] Manual live checklist for the user (see the report).

### Task 5: Docs, real stills, commit

- [ ] Re-run the pipeline on the 4 stills into `out/` (layers and scenes), then start the editor with `--no-browser`. Smoke-test with curl: list, scene, layer bytes.
- [ ] Update `README.md` (the editor usage and keys), `SKILL.md` (modules and pipeline), and `MEMORY.md`, plus `ERRORS.md` if anything took more than 2 attempts. Set the spec status.
- [ ] Run the full pytest and node suites, then make one commit.
