"""The overlay editor: a small local web server for drawing analysis graphics
on camera views while narrating (see the overlay-editor spec).

    python -m footsight.edit <folder-or-camera-view> [--port 8765] [--no-browser]

It listens on 127.0.0.1 only and serves nothing but the editor's own files
and the files of the stills it found. Nothing is written unless the user
exports a PNG or saves their overlays.
"""
import argparse
import json
import queue
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

EDITOR_DIR = Path(__file__).resolve().parent / "editor"
TAG_FONT = Path.home() / "Library" / "Fonts" / "Kanit-SemiBold.ttf"
DEFAULT_PORT = 8765
MAX_EXPORT_BYTES = 50 * 1024 * 1024
MAX_OVERLAYS_BYTES = 5 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
OVERLAY_TYPES = {"ring", "tag", "arrow", "link", "line", "zone"}
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".png": "image/png",
    ".ttf": "font/ttf",
}


@dataclass
class Still:
    """One generated camera view and the files around it, all named after
    the camera view: <name>.png, <name>_background.png, ..."""

    name: str
    folder: Path
    status: str = "ready"  # the studio adds: queued, processing, failed
    error: str | None = None
    version: int = 0  # bumped on every (re-)render, so the editor reloads images

    def path(self, suffix: str) -> Path:
        return self.folder / f"{self.name}{suffix}"

    @property
    def scene(self) -> Path:
        return self.path("_scene.json")

    @property
    def background(self) -> Path:
        return self.path("_background.png")

    @property
    def figures(self) -> Path:
        return self.path("_figures.png")

    @property
    def overlays(self) -> Path:
        return self.path("_overlays.json")

    @property
    def edited(self) -> Path:
        return self.path("_edited.png")


def find_stills(path) -> tuple[list[Still], list[str]]:
    """Stills ready for the editor, sorted by name (the order ←/→ steps
    through), and the names of ones missing a layer -- generated before the
    editor existed, so they need the pipeline re-run."""
    path = Path(path)
    if path.is_file():
        name = path.name.removesuffix("_scene.json") if path.name.endswith("_scene.json") else path.stem
        candidates = [Still(name, path.parent)]
    else:
        candidates = [Still(scene.name.removesuffix("_scene.json"), path) for scene in sorted(path.glob("*_scene.json"))]
    stills, skipped = [], []
    for still in candidates:
        complete = all(p.exists() for p in (still.scene, still.background, still.figures))
        (stills if complete else skipped).append(still if complete else still.name)
    return stills, skipped


def still_info(index: int, still: Still) -> dict:
    return {
        "id": index,
        "name": still.name,
        "has_overlays": still.overlays.exists(),
        "status": still.status,
        "error": still.error,
        "version": still.version,
    }


def _valid_overlays(data) -> bool:
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("overlays"), list):
        return False
    return all(isinstance(o, dict) and o.get("type") in OVERLAY_TYPES for o in data["overlays"])


def make_server(stills: list[Still], skipped: list[str], port: int = DEFAULT_PORT, studio=None,
                fallback_port: bool = True) -> ThreadingHTTPServer:
    """With a studio (footsight.studio), the server also takes captures from
    the Chrome extension, applies editor fixes and streams live still events."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet while recording
            pass

        def _send(self, status, body=b"", content_type="text/plain; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path):
            if not path.is_file():
                return self._send(404, b"Not found")
            self._send(200, path.read_bytes(), CONTENT_TYPES.get(path.suffix, "application/octet-stream"))

        def _still(self, parts):
            try:
                return stills[int(parts[2])]
            except (ValueError, IndexError):
                return None

        def do_GET(self):
            if studio is not None:
                studio.touch()
            path = unquote(urlparse(self.path).path)
            if path == "/":
                return self._send_file(EDITOR_DIR / "index.html")
            if path == "/editor/fonts/kanit-semibold.ttf":
                return self._send_file(TAG_FONT)
            if path.startswith("/editor/"):
                target = (EDITOR_DIR / path.removeprefix("/editor/")).resolve()
                if EDITOR_DIR not in target.parents:
                    return self._send(404, b"Not found")
                return self._send_file(target)
            parts = path.strip("/").split("/")
            if parts == ["api", "stills"]:
                listing = {
                    "stills": [still_info(i, s) for i, s in enumerate(stills)],
                    "skipped": skipped,
                    "studio": studio is not None,
                }
                return self._send(200, json.dumps(listing).encode(), "application/json")
            if parts == ["api", "events"] and studio is not None:
                return self._stream_events()
            if len(parts) == 4 and parts[:2] == ["api", "stills"] and parts[3] == "corrections" and studio is not None:
                still = self._still(parts)
                if still is None:
                    return self._send(404, b"Not found")
                return self._send(200, json.dumps(studio.corrections(still)).encode(), "application/json")
            if len(parts) == 4 and parts[:2] == ["api", "stills"]:
                still = self._still(parts)
                files = {"scene": "scene", "background": "background", "figures": "figures", "overlays": "overlays"}
                if still is None or parts[3] not in files:
                    return self._send(404, b"Not found")
                return self._send_file(getattr(still, files[parts[3]]))
            self._send(404, b"Not found")

        def _stream_events(self):
            """Server-sent events: the editor hears about new, processed and
            fixed stills without reloading."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            subscriber = studio.subscribe()
            try:
                while True:
                    try:
                        event = subscriber.get(timeout=15)
                        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")  # keeps the connection open
                    self.wfile.flush()
            except OSError:
                pass  # the editor closed the page
            finally:
                studio.unsubscribe(subscriber)

        def _capture(self):
            """A frame from the Chrome extension. Web pages can't send the custom
            header to another origin, and the Origin must be an extension, so
            only footsight's extension can drop stills in."""
            origin = self.headers.get("Origin") or ""
            if self.headers.get("X-Footsight-Capture") != "1" or not origin.startswith("chrome-extension://"):
                return self._send(403, b"Captures come from the footsight Chrome extension")
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_EXPORT_BYTES:
                return self._send(413, b"Too large")
            body = self.rfile.read(length)
            if not body.startswith(PNG_SIGNATURE):
                return self._send(400, b"A capture must be a PNG")
            try:
                meta = json.loads(unquote(self.headers.get("X-Footsight-Meta") or "{}"))
            except ValueError:
                meta = {}
            still = studio.capture(body, meta if isinstance(meta, dict) else {})
            self._send(200, json.dumps({"id": stills.index(still), "name": still.name}).encode(), "application/json")

        def _correct(self, still):
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_OVERLAYS_BYTES:
                return self._send(413, b"Too large")
            try:
                data = json.loads(self.rfile.read(length))
                from footsight.studio import validate_corrections
                validate_corrections(data)
            except (ValueError, UnicodeDecodeError) as error:
                return self._send(400, f"Invalid corrections: {error}".encode())
            try:
                fixes = studio.correct(still, data)
            except Exception as error:  # the re-render failed: keep the previous images
                return self._send(500, f"Couldn't apply the fix: {error}".encode())
            self._send(200, json.dumps({"version": still.version, "corrections": fixes}).encode(), "application/json")

        def do_POST(self):
            if studio is not None:
                studio.touch()
            parts = unquote(urlparse(self.path).path).strip("/").split("/")
            if parts == ["api", "capture"] and studio is not None:
                return self._capture()
            if len(parts) == 4 and parts[:2] == ["api", "stills"] and parts[3] == "corrections" and studio is not None:
                still = self._still(parts)
                return self._send(404, b"Not found") if still is None else self._correct(still)
            if not (len(parts) == 4 and parts[:2] == ["api", "stills"] and parts[3] in ("overlays", "export")):
                return self._send(404, b"Not found")
            still = self._still(parts)
            if still is None:
                return self._send(404, b"Not found")
            length = int(self.headers.get("Content-Length") or 0)
            limit = MAX_EXPORT_BYTES if parts[3] == "export" else MAX_OVERLAYS_BYTES
            if length > limit:
                return self._send(413, b"Too large")
            body = self.rfile.read(length)

            if parts[3] == "export":
                if not body.startswith(PNG_SIGNATURE):
                    return self._send(400, b"Export must be a PNG")
                still.edited.write_bytes(body)
                return self._send(200, json.dumps({"saved": still.edited.name}).encode(), "application/json")

            try:
                data = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                return self._send(400, b"Overlays must be JSON")
            if not _valid_overlays(data):
                return self._send(400, b"Not a footsight overlays file")
            still.overlays.write_text(json.dumps(data, indent=1))
            self._send(200, json.dumps({"saved": still.overlays.name}).encode(), "application/json")

    ThreadingHTTPServer.daemon_threads = True  # open event streams must not block shutdown
    try:
        return ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError:
        if not fallback_port:
            raise
        return ThreadingHTTPServer(("127.0.0.1", 0), Handler)  # port taken: any free one


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw analysis overlays on camera views, live.")
    parser.add_argument("path", help="A folder of generated stills, or one *_camera.png")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="Don't open the browser automatically")
    args = parser.parse_args()

    stills, skipped = find_stills(args.path)
    if not stills:
        raise SystemExit(
            f"No editable stills in {args.path}. Run the pipeline on your stills first "
            "(it writes *_camera_background.png, *_camera_figures.png and *_camera_scene.json)."
        )
    for name in skipped:
        print(f"Skipping {name}: made before the editor existed -- re-run the pipeline on it.")

    server = make_server(stills, skipped, args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"footsight editor: {url}  ({len(stills)} still(s); Ctrl-C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
