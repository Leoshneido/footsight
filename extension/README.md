# footsight Chrome extension

Captures the current video frame on the page you're watching and sends it to
footsight studio, which analyzes it and shows it in the editor.

## Set it up (once)

1. In Terminal, inside the footsight folder:
   `.venv/bin/python -m footsight.setup_chrome` — registers the footsight
   helper with Chrome, so the extension can start footsight itself.
2. Chrome → `chrome://extensions` → turn on **Developer mode** → **Load
   unpacked** → this `extension/` folder (or click the reload arrow if it's
   already loaded). The ID should read `kpjoeofameimecgoapeepcgbacpbkfai`.
3. Shortcut: **⌘⇧S** on Mac (Alt+Shift+S elsewhere). If the toolbar button
   shows **!**, Chrome couldn't assign it: set one at
   `chrome://extensions/shortcuts`.

## Use it

1. Click the footsight icon → name the project → **Choose location & start…**
   (Finder asks where) or **Start in <last location>**. footsight starts in
   the background and the editor opens when it's ready.
2. On the match page, pause on a moment and press ⌘⇧S → "Sent to footsight ✓".
   The still appears in the editor once it's processed.
3. The popup also reopens recent projects, captures ("Capture now"), and
   stops footsight. footsight stops by itself after 2 hours without use
   (never while the editor is open).

⌘⇧S with footsight off starts your last project and sends the frame to it.
With no project yet, the frame is kept and the popup opens: start a project
and the frame goes into it. If the helper isn't set up, or footsight takes
more than 60 s to start, the frame is saved to `Downloads/footsight-captures/`.
If the page's video is protected (DRM), both capture methods come out black
and nothing is saved — the extension doesn't try to get around protection.

Each project folder holds `footsight.log`, which is the first place to look
if footsight won't start. Remove the helper with
`.venv/bin/python -m footsight.setup_chrome --uninstall`.

Tests: `node --test extension/tests/*.test.mjs`.
