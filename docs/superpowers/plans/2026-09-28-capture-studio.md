# Capture Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A ⌘⇧S Chrome extension sends full-resolution frames to `python -m footsight.studio`. The studio processes them automatically into the live editor, where X, N and O fix players and the ball in about 2-3 s.

**Architecture:**
- `pipeline` splits into a cached `analyze` (slow) and a `render_still` that applies corrections (fast).
- `studio.py` runs a queue worker and adds capture, corrections and events routes to the `edit.py` server.
- The editor gains fix tools and live still states.
- `extension/` is an unpacked Manifest V3 extension whose pure core is tested with node.

**Tech Stack:** Python 3.14 stdlib (`http.server`, `threading`, `queue`), existing footsight modules, plain JavaScript, Chrome MV3, pytest, `node --test`.

**Spec:** `docs/superpowers/specs/2026-09-28-capture-studio-design.md`

## Global Constraints

- Detection IDs are stable (the detector's output order); added players are 1000+.
- Corrections format:
  `{removed: [id], added: [{id, feet: [x, y] still px, category}], ball: {mode: auto|set|none, pixel?}}`.
- The analysis cache is `<out>_analysis.json`; corrections are `<out>_corrections.json`.
- Capture requests need the `X-Footsight-Capture: 1` header and an `Origin` starting with `chrome-extension://`, carry a PNG ≤ 50 MB, and put metadata in the `X-Footsight-Meta` header (JSON).
- Session folder: `captures/<YYYY-MM-DD HH-MM> <name>/`, git-ignored; stills are `NNN.png`.
- Shortcut: `Command+Shift+S` on Mac, `Alt+Shift+S` by default.
- `run()` keeps its signature and outputs, including the review callbacks.
- Test-first. Commit only with the user's OK (not pre-approved for this build).
- Tests: `.venv/bin/python -m pytest -q`, `node --test footsight/editor/tests/*.test.mjs extension/tests/*.test.mjs`.

---

### Task 1: Pipeline split and corrections
**Files:** `footsight/pipeline.py`, `footsight/camera_view.py` (scene `id`, `scale` and `kits`), `tests/test_pipeline.py`, `tests/test_camera_view.py`.
- [ ] Failing tests:
  - `analyze` returns a JSON-able dict with stable IDs and poses keyed by ID (models mocked);
  - `render_still` with no corrections matches `run`'s calls;
  - a removed ID is excluded before the team split;
  - an added player is rendered in the chosen category with a box built from the vertical scale, and is not clustered;
  - ball set, none and auto;
  - the scene carries stable IDs, `scale` and `kits`;
  - `run` keeps working (the existing tests stay green).
- [ ] Implement.

### Task 2: Studio and server routes
**Files:** create `footsight/studio.py` and `tests/test_studio.py`; modify `footsight/edit.py`.
- [ ] Failing tests (analyze and render stubbed):
  - capture validation (header, origin, PNG, size);
  - numbering `001`, `002`;
  - queued, then processing, then ready;
  - failure recorded;
  - GET and POST corrections (validation, render called, version bump);
  - events stream lines;
  - `/api/stills` has `studio`, `status` and `version`;
  - `--open` re-registration.
- [ ] Implement, plus a `main()` that loads the models once.

### Task 3: Editor fix tools and live states
**Files:** `footsight/editor/editor.js`, `index.html`, `editor.css`, `geometry.js` (+ tests).
- [ ] Failing node tests: `imageToStill(px, scale)`, `nextCorrections(corrections, action)` (remove, add, ball set/none, removing an added player), `dropOverlaysFor(overlays, ids)`.
- [ ] Implement:
  - the X, N and O tools, the chooser, the Updating state;
  - undo including corrections;
  - EventSource, placeholders, versioned reloads;
  - fix tools hidden without the studio.

### Task 4: Chrome extension
**Files:** create `extension/manifest.json`, `background.js`, `capture-core.js`, `package.json`, `tests/capture-core.test.mjs`, `README.md`.
- [ ] Failing node tests for `isMostlyBlack`, `captureHeaders`, `fallbackFilename` and `describeResult`.
- [ ] Implement; `node --check` on `background.js`.

### Task 5: Real run, docs, log
- [ ] Start the studio on real models. POST the 4 stills as captures via curl with extension headers, then check processing to ready.
- [ ] Apply an "add player" correction on the Bayern still and remove the watermark on 12.00.14 through the API; check the images.
- [ ] Take a headless screenshot of the editor.
- [ ] Update README, SKILL.md, MEMORY.md (and ERRORS.md if needed) and `.gitignore` (`captures/`).
- [ ] Give the user a manual checklist and ask to commit.
