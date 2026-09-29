# Projects from the Chrome Extension — Design Spec

Date: 2026-09-29
Status: Approved section by section in chat; built straight through at the user's request

## Context

Today the user starts the studio from Terminal (`python -m footsight.studio`).
The goal is to kick everything off from the extension:

1. Click the footsight icon and start a new project.
2. Choose where it lives with the Mac folder picker.
3. From then on, ⌘⇧S drops stills into that project; they are processed and
   appear in the editor.

A Chrome extension can't start programs or see real folder paths, so a small
footsight **helper**, started by Chrome on demand (native messaging),
handles the Mac side.

## Decisions (chat, 2026-09-28/29)

- **Startup:** Chrome starts footsight on demand. Rejected: an always-on
  login item, which keeps the models in memory all the time; and a manual
  Terminal start, which isn't one-click.
- **Location:** the Mac "Choose folder" dialog; the last-used location is
  the default. Rejected: a fixed home folder, and typed paths.
- **Helper structure:** the helper starts the studio as its own background
  process. Rejected: the studio living inside the native-messaging
  connection, because Chrome sleeps the extension's service worker and the
  connection drops.
- **⌘⇧S with the studio off:**
  - The frame is grabbed immediately.
  - If there is a recent project, it is auto-started (about 10 s) and the
    frame is delivered.
  - If no project exists yet, the frame is kept and the popup opens for a
    new project, which receives it.
  - If the helper isn't set up, the frame goes to Downloads.
  - If auto-start takes more than 60 s, the frame goes to Downloads.
- **Clicking the icon** now opens a popup; it no longer captures directly.
  ⌘⇧S and the popup's "Capture now" capture.
- **Changed during the build:** choosing the folder and starting are one step
  (`new_project` with `choose: true`), run by `background.js`. Chrome closes
  the popup as soon as the Finder dialog takes focus, so a separate
  `choose_folder` request followed by Start from the popup can't work. The
  popup offers "Start in <last location>" and "Choose location & start…".

## Architecture

```
extension popup ──sendNativeMessage──▶ footsight helper (footsight/host.py, one request per run)
   │                                      ├─ new_project (choose: osascript "choose folder") / open: create folder, start the studio (detached), record it
   │                                      ├─ status / recent / stop
   │                                      └─ ~/.footsight/projects.json (recent, last location), studio.json (pid, project)
   ▼
⌘⇧S (background.js) ──POST /api/capture──▶ footsight studio (python -m footsight.studio --open DIR --no-browser --idle-minutes 120)
```

### Helper protocol

Chrome native messaging: a 4-byte little-endian length, then UTF-8 JSON.

| Request | Response |
|---|---|
| `{"cmd":"status"}` | `{"ok":true, "running":bool, "starting":bool, "project":{"name","path"}\|null, "stills":{"ready","processing","queued","failed"}, "url"}`, plus `"error"` with the log tail if the studio died |
| `{"cmd":"new_project","name","location"?,"choose"?:bool}` | `{"ok":true,"path","existing":bool,"starting":true}`, `{"ok":false,"cancelled":true}` (Finder dialog cancelled), or `{"ok":false,"error"}` |
| `{"cmd":"open","path"}` | `{"ok":true,"path","starting":true}` |
| `{"cmd":"recent"}` | `{"ok":true,"projects":[{"name","path","stills"}],"last_location"}` |
| `{"cmd":"stop"}` | `{"ok":true,"stopped":bool}` |

**Starting the studio:** `Popen([python, "-m", "footsight.studio",
"--open", DIR, "--no-browser", "--idle-minutes", "120"])`.
- The working directory is the repo root, because the weights paths are
  relative.
- It starts in its own session, so it outlives the helper.
- stdout and stderr go to `DIR/footsight.log`.
- The helper writes `studio.json` with `{pid, project}`.

**Running check:** the recorded pid is alive AND `GET /api/stills` answers.

**Stop:** SIGTERM (the studio handles it and shuts down cleanly), wait up to
5 s, then remove `studio.json`.

**Projects:**
- A folder is a footsight project if it has an `originals/` folder.
- New project at `<location>/<name>`:
  - if it exists and is a project, it is reopened (`existing: true`);
  - if it exists and isn't one, `<name> (2)`, `(3)`, … is used;
  - an unwritable location gives an error.
- Recent: 8 entries, newest first; missing folders are skipped.

The state folder is `~/.footsight` (the `FOOTSIGHT_HOME` environment
variable overrides it for tests).

### Studio additions

- **Idle stop:** `--idle-minutes N`. Every HTTP request, capture and fix
  resets the idle clock. An open editor (an event subscriber) counts as
  active. After N idle minutes the server shuts down.
- **SIGTERM** triggers a clean shutdown.

### Setup (`python -m footsight.setup_chrome [--uninstall]`)

- **Launcher:** writes `~/.footsight/footsight-host`, an executable
  `#!/bin/sh` that runs `exec <venv python> -m footsight.host`.
- **Registration:** writes
  `~/Library/Application Support/Google/Chrome/NativeMessagingHosts/com.footsight.host.json`
  with `{name, description, path, type: "stdio", allowed_origins:
  ["chrome-extension://<ID>/"]}`.
- **Extension ID:** computed from the `key` in `extension/manifest.json`:
  the SHA-256 of the DER public key, first 32 hex digits mapped 0-f to a-p.
- It prints the next steps (reload the extension). `--uninstall` removes
  the registration.

The key pair is generated once; only the public key goes in the manifest.

### Extension

- **Manifest:** adds `key`, the `nativeMessaging` and `storage`
  permissions, and `action.default_popup: popup.html`.
- **`popup.html` / `popup.js`:** states `setup | off | starting | running`
  (see the design in chat), polling status every second while starting.
  Its parts:
  - the New project form (name defaulting to "Match YYYY-MM-DD",
    "Start in <last location>", "Choose location & start…");
  - Recent;
  - Open editor (focuses an existing editor tab if there is one);
  - Capture now;
  - Stop footsight.
- **`background.js`:** ⌘⇧S grabs the frame, then `deliverCapture`. A
  pending frame waiting for a new project is kept in
  `chrome.storage.session`; `background.js` delivers it once the new
  project's studio answers (the popup may already be closed).
- **`capture-core.js`** (pure, node-tested) adds:
  - `popupState(reply, error)`;
  - `defaultProjectName(date)`;
  - `deliverCapture(frame, deps)`, the auto-start sequence with injected
    Chrome and helper calls.

## Errors

| Case | Behavior |
|---|---|
| Helper not registered | The popup shows the setup instructions; ⌘⇧S saves to Downloads |
| Folder picker cancelled | Nothing happens |
| Location not writable | Error message; pick again |
| Studio fails to start (missing model, port 8765 in use) | The popup shows the error and the last lines of `footsight.log` |
| Auto-start takes over 60 s | The frame goes to Downloads; toast |
| Project folder moved or deleted | Dropped from Recent |

## Testing

- **pytest:**
  - message framing;
  - each helper command, with the process launcher, folder picker and
    HTTP check stubbed;
  - name clashes;
  - recent-projects persistence and skipping missing folders;
  - a stale `studio.json` (dead pid);
  - setup writing the registration and launcher to temporary paths, and
    the extension ID derivation (checked against a known key/ID pair);
  - the studio's idle stop.
- **node:** `popupState`, `defaultProjectName`, and the `deliverCapture`
  branches (studio up; auto-start then send; no project, so the frame is
  kept; helper missing, so Downloads; timeout, so Downloads).
- **Manual (user):** setup, new project through the Finder picker, ⌘⇧S with
  the studio off, reopening a recent project, Stop.

## Out of scope

Windows and Linux, several studios at once, and publishing the extension.
