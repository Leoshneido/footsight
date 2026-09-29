# Capture Studio + Chrome Extension — Design Spec

Date: 2026-09-28
Status: Approved and implemented (plan: docs/superpowers/plans/2026-09-28-capture-studio.md)

## Context

The user watches matches on a website and narrates over stills. A spike
(MEMORY.md, 2026-09-27) proved that on their site a Chrome extension can
read the `<video>` frame at full resolution. This adds:

- a **Chrome extension** that captures the current frame with ⌘⇧S and sends
  it to footsight;
- a **studio** (`python -m footsight.studio`): the editor server plus a
  capture inbox. It processes each still automatically, with no pop-ups, and
  new stills appear in the open editor;
- **fix tools in the editor**: remove a player (X), add a missed player (N,
  e.g. hidden by an overlap on the Bayern still), and place, move or remove
  the ball (O). Each fix redoes the still in about 2-3 s.

Paid services such as Paramount+ stay manual. Nothing bypasses DRM or copy
protection.

## Decisions (chat, 2026-09-27/28)

- **After capture:** send straight to footsight, which is already running.
  The rejected alternatives were a watched folder (slower, no feedback) and
  Chrome native messaging (fiddly setup).
- **Review:** processing is automatic, and fixes happen in the editor. The
  pop-up review windows are rejected for this flow; the standalone
  `pipeline` command keeps them.
- **Shortcut:** ⌘⇧S on Mac (Alt+Shift+S elsewhere), changeable at
  chrome://extensions/shortcuts. The toolbar button always works.
- **Added players:**
  - The user places the feet and picks a side (team A or B in their actual
    kit colors, goalkeeper, or referee).
  - The box is 1.8 m tall at that spot (the player-height scale), 0.4 of
    that wide.
  - Drawn as the standing figure; not part of the team split.

## Architecture

```
Chrome extension (extension/, loaded unpacked)
  ⌘⇧S -> read the full-resolution <video> frame (screenshot fallback; says so if both are black)
      -> POST http://127.0.0.1:8765/api/capture   (X-Footsight-Capture header, chrome-extension:// origin)
      -> toast "Sent to footsight ✓ (still N)"; if the studio is unreachable,
         save to Downloads/footsight-captures instead

python -m footsight.studio [--session NAME | --open DIR] [--port 8765] [--no-browser]
  loads the models once; session folder captures/<YYYY-MM-DD HH-MM> <name>/ (git-ignored)
  capture -> NNN.png saved -> queued -> worker: pipeline.analyze (slow, cached) -> pipeline.render_still (fast)
  editor (same server) gets live events (server-sent events) as stills are queued, processed, ready or failed
  fixes: POST corrections -> render only (about 2-3 s) -> "updated" event
```

### Pipeline split (`footsight/pipeline.py`)

- **`analyze(input_path, detection_model, weights_kp, weights_line,
  pose_model=None) -> dict`**. This is the slow step:
  - calibration;
  - detection, with **stable IDs** (each detection's position in the
    detector's output);
  - poses of the on-pitch people;
  - the automatic ball's ground pixel.

  It is JSON-serializable and saved as `<out>_analysis.json`:
  `{version, image: [w, h], homography, detections: [{id, box, role}],
  poses: {id: 17x3 | null}, ball_pixel}`.
- **`render_still(input_path, analysis, output_path, corrections=None)` (not `render`: that name is the renderer module)**. This is
  the fast step:
  - applies the corrections;
  - off-pitch filter;
  - team split on on-pitch `player` detections only (added players are
    excluded);
  - kit colors;
  - the top-down mockup and the camera view, with its layers and scene.
- **`run(...)`**: unchanged signature and behavior. It is analyze, then
  review callbacks turned into corrections, then render.
- **Corrections** (`<out>_corrections.json`):
  `{removed: [id], added: [{id, feet: [x, y] still px, category}],
  ball: {mode: "auto"|"set"|"none", pixel?: [x, y] still px}}`. Added IDs
  start at 1000.
- **Scene additions:**
  - players carry their stable `id`;
  - `scale` (1.5);
  - `kits`: category -> {shirt, shorts}, for the chooser.

### Studio (`footsight/studio.py`) and server (`footsight/edit.py`)

- `edit.make_server(stills, skipped, port, studio=None)`. Without a studio,
  the editor behaves as today and the fix tools are hidden. With one:
  - `POST /api/capture`: requires the `X-Footsight-Capture: 1` header and an
    `Origin: chrome-extension://…`; the body is a PNG ≤ 50 MB; metadata goes
    in the `X-Footsight-Meta` header (JSON: page title, host, video time,
    captured at). Replies `{id, name}`.
  - `GET /api/events`: server-sent events `{type: "still", still}`.
  - `GET` and `POST /api/stills/<id>/corrections`: POST is validated, then
    rendered synchronously; replies with the new scene version.
  - `/api/stills` adds `studio: true` and, per still, `status`, `error` and
    `version`.
- **`Studio`:**
  - takes the session dir and the analyze and render functions (injected, so
    tests stub them);
  - holds a thread-safe still list shared with the server;
  - has one worker thread and a FIFO queue;
  - notifies event subscribers;
  - `--open` re-registers a past session's stills: ready when the analysis
    exists, otherwise queued.
- **Port busy:** exit with a message (the extension expects 8765).

### Editor

- **Fix tools**, only when `studio: true`:
  - **X** remove the clicked player;
  - **N** add a player: click the feet, then a chooser pops up with the two
    teams' kit swatches, goalkeeper and referee (keys 1-4 or click);
  - **O** click to place or move the ball; Shift+click to remove it.
- **Fixes and undo:** fixes are part of the undo history (a snapshot of
  overlays plus corrections). Undo or redo re-posts the corrections when
  they differ.
- **Drawings on a removed player** are dropped in the same action.
- **"Updating…"** shows while a fix renders; the images reload with `?v=`.
- **Still states:** ←/→ includes queued, processing and failed stills,
  shown on the stage as "Processing… / Failed: reason" placeholders.
  EventSource updates the list, and the current still refreshes when its
  version changes (drawings kept).

### Extension (`extension/`)

- Manifest V3.
- Permissions: `activeTab`, `scripting`, `downloads`; host permission
  `http://127.0.0.1:8765/*`.
- The `capture` command's suggested key is `Command+Shift+S` on Mac and
  `Alt+Shift+S` by default. On install, if Chrome didn't assign it, the
  toolbar badge shows "!" and the title points to
  chrome://extensions/shortcuts.
- `capture-core.js` holds the pure pieces, tested with `node --test`:
  - `isMostlyBlack(pixels)`;
  - `captureHeaders(meta)`;
  - `fallbackFilename(date, method)`;
  - `describeResult(...)`.
- `background.js` (a module service worker) is the Chrome glue.

## Errors

| Case | Behavior |
|---|---|
| Calibration fails, close-up, replay | The still is marked failed with the reason; the original is kept. |
| Studio not running | The extension saves to Downloads/footsight-captures and says so. |
| Both capture methods black | The extension says the video is protected and saves nothing. |
| Invalid capture request | 400 or 403 (missing header, non-extension origin, non-PNG), 413 if too large. |
| A fix fails | Editor notice; the previous images stay. |

## Testing (test-first)

- **pytest:**
  - analyze/render split: `run` produces identical files to before;
  - corrections: remove (player gone, team split redone), add (chosen kit,
    not clustered, box from the vertical scale), ball set/none/auto;
  - IDs stable across corrections;
  - studio: capture validation, numbering, queue to ready, failure reason,
    corrections flow, events, `--open` of an existing session;
  - models stubbed throughout.
- **node:** extension core; editor helpers for turning a click into still
  pixels and building corrections.
- **Manual (user):**
  - loading the extension, ⌘⇧S on the site;
  - stills appearing live in the editor;
  - X, N, O, including the Bayern missing player;
  - `--open`.

## Out of scope

Burst capture, capturing on paid services (manual), publishing the
extension, measurements, and video playback inside footsight.
