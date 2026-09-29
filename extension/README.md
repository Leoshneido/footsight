# footsight Chrome extension

Captures the current video frame on the page you're watching and sends it to
footsight studio, which analyzes it and shows it in the editor.

## Load it (once)

1. Chrome → `chrome://extensions` → turn on **Developer mode**.
2. **Load unpacked** → choose this `extension/` folder.
3. Shortcut: **⌘⇧S** on Mac (Alt+Shift+S elsewhere). If the toolbar button
   shows **!**, Chrome couldn't assign it (another shortcut uses it): set one at
   `chrome://extensions/shortcuts`. Clicking the toolbar button always captures.

## Use it

1. Start the studio: `python -m footsight.studio --session "Barca v Feyenoord"`.
2. On the match page, pause on a moment and press ⌘⇧S → "Sent to footsight ✓".
3. The still appears in the editor as soon as it's processed.

If the studio isn't running, the still is saved to
`Downloads/footsight-captures/` instead. If the page's video is protected
(DRM), both capture methods come out black and nothing is saved — the
extension doesn't try to get around protection.

Tests: `node --test extension/tests/*.test.mjs`.
