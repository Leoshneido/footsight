"""The overlay editor: a small local web server for drawing analysis graphics
on camera views while narrating (see the overlay-editor spec).

    python -m footsight.edit <folder-or-camera-view> [--port 8765] [--no-browser]

It listens on 127.0.0.1 only and serves nothing but the editor's own files
and the files of the stills it found. Nothing is written unless the user
exports a PNG or saves their overlays.
"""
import argparse
import json
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


def _valid_overlays(data) -> bool:
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("overlays"), list):
        return False
    return all(isinstance(o, dict) and o.get("type") in OVERLAY_TYPES for o in data["overlays"])


def make_server(stills: list[Still], skipped: list[str], port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
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
                    "stills": [{"id": i, "name": s.name, "has_overlays": s.overlays.exists()} for i, s in enumerate(stills)],
                    "skipped": skipped,
                }
                return self._send(200, json.dumps(listing).encode(), "application/json")
            if len(parts) == 4 and parts[:2] == ["api", "stills"]:
                still = self._still(parts)
                files = {"scene": "scene", "background": "background", "figures": "figures", "overlays": "overlays"}
                if still is None or parts[3] not in files:
                    return self._send(404, b"Not found")
                return self._send_file(getattr(still, files[parts[3]]))
            self._send(404, b"Not found")

        def do_POST(self):
            parts = unquote(urlparse(self.path).path).strip("/").split("/")
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

    try:
        return ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError:
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
