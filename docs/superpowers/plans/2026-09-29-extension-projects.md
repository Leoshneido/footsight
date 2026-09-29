# Extension Projects Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start, reopen and stop footsight projects from the extension popup, with the Mac folder picker. ⌘⇧S auto-starts the last project when footsight is off.

**Architecture:**
- `footsight/projects.py`: pure project and state bookkeeping.
- `footsight/host.py`: the native-messaging helper; it launches the studio as a detached process.
- `footsight/setup_chrome.py`: registers the helper with Chrome.
- Studio: idle stop and SIGTERM handling.
- Extension: a popup, and a `deliverCapture` auto-start sequence in `capture-core.js`.

**Tech Stack:** Python 3.14 stdlib (`struct`, `subprocess`, `signal`, `urllib`), macOS `osascript`, OpenSSL (key, one-off), Chrome MV3 native messaging, pytest, `node --test`.

**Spec:** `docs/superpowers/specs/2026-09-29-extension-projects-design.md`

## Global Constraints

- The host name is `com.footsight.host`.
- The state folder is `~/.footsight` (`FOOTSIGHT_HOME` overrides it).
- A project is a folder with `originals/`. Recent keeps 8 entries.
- The studio starts with `--open DIR --no-browser --idle-minutes 120`, cwd at the repo root, logging to `DIR/footsight.log`.
- Auto-start waits at most 60 s, then falls back to Downloads.
- Test-first. Commit only with the user's OK.
- Tests: `.venv/bin/python -m pytest -q` and `node --test footsight/editor/tests/*.test.mjs extension/tests/*.test.mjs`.

---

### Task 1: `projects.py`
State files, recent projects, `is_project`, `project_folder_for` (name clashes), the studio record and running check (injected HTTP probe).

- [ ] Failing tests.
- [ ] Implement.
- [ ] Green.

### Task 2: `host.py`
- `read_message` and `write_message`.
- `handle(request, env)` for status, choose_folder, new_project, open, recent and stop.
- `env` injects `launch_studio`, `choose_folder`, `probe` and `kill`.

- [ ] Failing tests (fake env).
- [ ] Implement, plus `main()`.
- [ ] Green.

### Task 3: Studio idle stop and SIGTERM
- `Studio.touch()` and `Studio.idle_seconds()`.
- The server touches the studio on every request.
- `idle_watchdog(studio, server, minutes)` shuts the server down when idle and nobody is subscribed.
- `--idle-minutes`; SIGTERM leads to a clean shutdown.

- [ ] Failing tests.
- [ ] Implement.
- [ ] Green.

### Task 4: `setup_chrome.py` and the extension key
- `extension_id(public_key_b64)`.
- `install(home, hosts_dir, python)` writes the launcher and the registration; `uninstall`.
- Generate the key pair once with openssl; put only the public key in `manifest.json`.

- [ ] Failing tests: a known key and ID pair; the files written.
- [ ] Implement.
- [ ] Green.

### Task 5: Extension popup and auto-start
- `capture-core.js`: `popupState`, `defaultProjectName`, `deliverCapture`.
- New files: `popup.html`, `popup.css`, `popup.js`.
- `background.js` wiring, and manifest permissions.

- [ ] Failing node tests.
- [ ] Implement.
- [ ] `node --check`.
- [ ] Green.

### Task 6: Real run and docs
- [ ] Run setup for real, then drive the helper directly (framed messages) through `new_project`, `status`, a capture via HTTP, and `stop`.
- [ ] Update README, `extension/README`, SKILL.md and MEMORY.md.
- [ ] Give the user the checklist, and ask to commit.
