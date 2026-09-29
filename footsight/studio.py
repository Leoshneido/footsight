"""The capture studio: the overlay editor plus an inbox for stills sent by
the footsight Chrome extension (see the capture-studio spec).

    python -m footsight.studio [--session "Barca v Feyenoord" | --open DIR] [--port 8765] [--no-browser]

Each capture is saved, queued and processed in the background with no
pop-ups, and appears in the open editor when ready. Fixes made in the editor
(remove or add a player, move the ball) re-render from the cached analysis
in a couple of seconds.
"""
import argparse
import datetime
import json
import queue
import re
import signal
import threading
import time
import webbrowser
from pathlib import Path

from footsight import edit
from footsight.edit import Still

CAPTURES_DIR = Path("captures")
CATEGORIES = {"team_a", "team_b", "goalkeeper", "referee"}
BALL_MODES = {"auto", "set", "none"}
ADDED_ID_START = 1000


def session_folder(root, name: str, now: datetime.datetime | None = None) -> Path:
    """captures/<YYYY-MM-DD HH-MM> <name>: sorts by date, readable in Finder."""
    stamp = (now or datetime.datetime.now()).strftime("%Y-%m-%d %H-%M")
    safe = re.sub(r'[\\/:*?"<>|]+', "-", name or "").strip()
    return Path(root) / (f"{stamp} {safe}" if safe else stamp)


def empty_corrections() -> dict:
    return {"removed": [], "added": [], "ball": {"mode": "auto"}, "sides": {}}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_corrections(data) -> dict:
    """Check a corrections payload from the editor and give new added
    players their ids (1000 and up). Raises ValueError when invalid."""
    if not isinstance(data, dict):
        raise ValueError("corrections must be an object")
    removed = data.get("removed", [])
    if not isinstance(removed, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in removed):
        raise ValueError("removed must be a list of player ids")

    added = data.get("added", [])
    if not isinstance(added, list):
        raise ValueError("added must be a list")
    taken = [a["id"] for a in added if isinstance(a, dict) and isinstance(a.get("id"), int)]
    next_id = max(taken + [ADDED_ID_START - 1]) + 1
    clean_added = []
    for player in added:
        feet = player.get("feet") if isinstance(player, dict) else None
        if not (isinstance(feet, list) and len(feet) == 2 and all(_is_number(v) for v in feet)):
            raise ValueError("an added player needs feet [x, y]")
        if player.get("category") not in CATEGORIES:
            raise ValueError(f"unknown category {player.get('category')!r}")
        player_id = player.get("id")
        if not isinstance(player_id, int) or isinstance(player_id, bool):
            player_id, next_id = next_id, next_id + 1
        clean_added.append({"id": player_id, "feet": [float(v) for v in feet], "category": player["category"]})

    ball = data.get("ball", {"mode": "auto"})
    if not isinstance(ball, dict) or ball.get("mode") not in BALL_MODES:
        raise ValueError("ball mode must be auto, set or none")
    clean_ball = {"mode": ball["mode"]}
    if ball["mode"] == "set":
        pixel = ball.get("pixel")
        if not (isinstance(pixel, list) and len(pixel) == 2 and all(_is_number(v) for v in pixel)):
            raise ValueError("a placed ball needs pixel [x, y]")
        clean_ball["pixel"] = [float(v) for v in pixel]
    sides = data.get("sides", {})
    if not isinstance(sides, dict):
        raise ValueError("sides must map player ids to sides")
    for player_id, side in sides.items():
        if not str(player_id).isdigit():
            raise ValueError(f"not a player id: {player_id!r}")
        if side not in CATEGORIES:
            raise ValueError(f"unknown side {side!r}")
    return {"removed": removed, "added": clean_added, "ball": clean_ball, "sides": {str(k): v for k, v in sides.items()}}


class Studio:
    """Session folder, still list (shared with the editor server), one
    background worker, and live events for the editor.

    analyze(original_path) -> analysis dict (the slow step);
    render(original_path, analysis, output_path, corrections) (the fast one).
    """

    def __init__(self, session_dir, analyze, render):
        self.dir = Path(session_dir)
        (self.dir / "originals").mkdir(parents=True, exist_ok=True)
        self.analyze, self.render = analyze, render
        self.stills: list[Still] = []
        self._lock = threading.Lock()
        self._render_lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._subscribers: list[queue.Queue] = []
        self._last_activity = time.monotonic()
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    @classmethod
    def open(cls, session_dir, analyze, render) -> "Studio":
        """Reopen a past session: processed stills come back ready, stills
        captured but never processed are queued."""
        studio = cls(session_dir, analyze, render)
        for original in sorted((studio.dir / "originals").glob("[0-9][0-9][0-9].png")):
            with studio._lock:
                still = studio._register(original.stem)
            done = all(p.exists() for p in (studio.analysis_path(still), still.scene, still.background, still.figures))
            if done:
                still.status, still.version = "ready", 1
            else:
                studio._queue.put(still)
        return studio

    # ---- paths (all named after the capture number, e.g. 001) ----

    @staticmethod
    def number(still: Still) -> str:
        return still.name.split("_")[0]

    def original_path(self, still: Still) -> Path:
        return self.dir / "originals" / f"{self.number(still)}.png"

    def output_path(self, still: Still) -> Path:
        return self.dir / f"{self.number(still)}.png"

    def analysis_path(self, still: Still) -> Path:
        return self.dir / f"{self.number(still)}_analysis.json"

    def corrections_path(self, still: Still) -> Path:
        return self.dir / f"{self.number(still)}_corrections.json"

    # ---- idle tracking (a studio started from the extension stops itself) ----

    def touch(self) -> None:
        self._last_activity = time.monotonic()

    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_activity

    def should_stop_for_idle(self, idle_minutes: float) -> bool:
        """Idle for longer than idle_minutes (0 = never) with no editor page
        open -- an open editor listens for events and counts as in use."""
        with self._lock:
            watched = bool(self._subscribers)
        return idle_minutes > 0 and not watched and self.idle_seconds() > idle_minutes * 60

    # ---- capture and processing ----

    def _register(self, number: str) -> Still:
        still = Still(f"{number}_camera", self.dir, status="queued")
        self.stills.append(still)
        return still

    def capture(self, png: bytes, meta: dict) -> Still:
        self.touch()
        with self._lock:
            number = f"{len(self.stills) + 1:03d}"
            (self.dir / "originals" / f"{number}.png").write_bytes(png)
            (self.dir / "originals" / f"{number}.json").write_text(json.dumps(meta or {}, indent=1))
            still = self._register(number)
        self._notify(still)
        self._queue.put(still)
        return still

    def _work(self) -> None:
        while True:
            still = self._queue.get()
            if still is None:
                return
            self._process(still)

    def _process(self, still: Still) -> None:
        still.status = "processing"
        self._notify(still)
        try:
            original = str(self.original_path(still))
            analysis = self.analyze(original)
            self.analysis_path(still).write_text(json.dumps(analysis))
            with self._render_lock:
                self.render(original, analysis, str(self.output_path(still)), self.corrections(still))
            still.status, still.error = "ready", None
            still.version += 1
        except Exception as error:  # a close-up or replay can't be calibrated: report, keep the original
            still.status, still.error = "failed", str(error) or error.__class__.__name__
        self._notify(still)

    def corrections(self, still: Still) -> dict:
        path = self.corrections_path(still)
        return json.loads(path.read_text()) if path.exists() else empty_corrections()

    def correct(self, still: Still, data) -> dict:
        """Apply the editor's fixes: re-render from the cached analysis (a
        couple of seconds, no models), then save them."""
        self.touch()
        fixes = validate_corrections(data)
        analysis = json.loads(self.analysis_path(still).read_text())
        with self._render_lock:
            self.render(str(self.original_path(still)), analysis, str(self.output_path(still)), fixes)
        self.corrections_path(still).write_text(json.dumps(fixes, indent=1))
        still.version += 1
        self._notify(still)
        return fixes

    # ---- live events ----

    def subscribe(self) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def _notify(self, still: Still) -> None:
        event = {"type": "still", "still": edit.still_info(self.stills.index(still), still)}
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.put(event)

    def stop(self) -> None:
        self._queue.put(None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture stills from Chrome and analyze them live.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--session", default="", help="A name for this session, e.g. the match")
    group.add_argument("--open", dest="open_dir", help="Reopen a past session folder")
    parser.add_argument("--port", type=int, default=edit.DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--idle-minutes", type=float, default=0,
                        help="Stop after this many idle minutes with no editor open (0 = never)")
    args = parser.parse_args()

    from footsight import pipeline, pitch_calibration, player_detection, pose

    print("Loading models…")
    pose_model = pose.load_model()
    detection_model = player_detection.load_model()
    kp, lines = str(pitch_calibration.DEFAULT_WEIGHTS_KP), str(pitch_calibration.DEFAULT_WEIGHTS_LINE)

    def analyze(path):
        return pipeline.analyze(path, detection_model, kp, lines, pose_model)

    if args.open_dir:
        studio = Studio.open(args.open_dir, analyze, pipeline.render_still)
    else:
        studio = Studio(session_folder(CAPTURES_DIR, args.session), analyze, pipeline.render_still)

    try:
        server = edit.make_server(studio.stills, [], args.port, studio=studio, fallback_port=False)
    except OSError:
        raise SystemExit(f"Port {args.port} is in use -- is another footsight studio or editor running? "
                         "The Chrome extension sends captures to this port.")
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Ready -- capture with ⌘⇧S in Chrome. Session: {studio.dir}")
    print(f"Editor: {url}  (Ctrl-C to stop)")
    if not args.no_browser:
        webbrowser.open(url)

    def shut_down(*_):  # SIGTERM (Stop footsight) or idle: stop serving, then clean up below
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shut_down)

    def watch_idle():
        while True:
            time.sleep(30)
            if studio.should_stop_for_idle(args.idle_minutes):
                print(f"Idle for {args.idle_minutes:g} minutes -- stopping.")
                return shut_down()

    threading.Thread(target=watch_idle, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        studio.stop()
        server.server_close()


if __name__ == "__main__":
    main()
