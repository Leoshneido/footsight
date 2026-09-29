"""The footsight helper Chrome starts on demand (native messaging host
com.footsight.host). It answers one request from the extension and exits;
the studio it starts runs on as its own background process.

Requests (see the extension-projects spec): status, new_project, open,
recent, stop. Registered with Chrome by `python -m footsight.setup_chrome`.
"""
import json
import signal
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from footsight import projects

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDIO_URL = "http://127.0.0.1:8765/"
IDLE_MINUTES = 120
STATUSES = ("ready", "processing", "queued", "failed")


# ---- Chrome native messaging: 4-byte little-endian length + UTF-8 JSON ----

def read_message(stream) -> dict | None:
    header = stream.read(4)
    if len(header) < 4:
        return None
    (length,) = struct.unpack("<I", header)
    return json.loads(stream.read(length).decode("utf-8"))


def write_message(stream, message: dict) -> None:
    body = json.dumps(message).encode("utf-8")
    stream.write(struct.pack("<I", len(body)))
    stream.write(body)
    stream.flush()


# ---- the Mac side, behind one object so tests can stand in for it ----

class MacEnv:
    def launch(self, folder: Path) -> int:
        """Start the studio on `folder` in its own session, so it outlives
        this helper. cwd is the repo: the model weights paths are relative."""
        log = open(Path(folder) / "footsight.log", "ab")
        process = subprocess.Popen(
            [sys.executable, "-m", "footsight.studio", "--open", str(folder), "--no-browser", "--idle-minutes", str(IDLE_MINUTES)],
            cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        return process.pid

    def choose(self, default) -> str | None:
        """The Finder "Choose folder" dialog; None if cancelled."""
        default_clause = f' default location (POSIX file "{default}")' if default and Path(default).is_dir() else ""
        script = [
            'tell application "System Events"',
            "activate",
            f'set chosen to choose folder with prompt "Where should the footsight project go?"{default_clause}',
            "end tell",
            "POSIX path of chosen",
        ]
        args = ["osascript"] + [part for line in script for part in ("-e", line)]
        result = subprocess.run(args, capture_output=True, text=True)
        return result.stdout.strip().rstrip("/") or None if result.returncode == 0 else None

    def alive(self, pid: int) -> bool:
        return projects.pid_alive(pid)

    def probe(self) -> bool:
        try:
            with urllib.request.urlopen(STUDIO_URL + "api/stills", timeout=1) as response:
                return response.status == 200
        except OSError:
            return False

    def kill(self, pid: int) -> None:
        try:
            import os

            os.kill(pid, signal.SIGTERM)  # the studio shuts down cleanly on SIGTERM
            for _ in range(50):
                if not self.alive(pid):
                    return
                time.sleep(0.1)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def summary(self) -> dict:
        try:
            with urllib.request.urlopen(STUDIO_URL + "api/stills", timeout=2) as response:
                stills = json.load(response)["stills"]
        except (OSError, ValueError, KeyError):
            return {}
        return {status: sum(1 for s in stills if s["status"] == status) for status in STATUSES}


# ---- requests ----

def _log_tail(folder, lines: int = 4) -> str:
    try:
        return "\n".join((Path(folder) / "footsight.log").read_text(errors="replace").strip().splitlines()[-lines:])
    except OSError:
        return ""


def _status(env) -> dict:
    record = projects._read("studio.json")
    reply = {"ok": True, "running": False, "starting": False, "project": None, "stills": {}, "url": STUDIO_URL}
    if not record.get("pid"):
        return reply
    folder = Path(record["project"])
    if not env.alive(record["pid"]):
        projects.clear_studio_record()
        tail = _log_tail(folder)
        if tail:
            reply["error"] = f"footsight stopped. Last lines of {folder / 'footsight.log'}:\n{tail}"
        return reply
    reply["project"] = {"name": folder.name, "path": str(folder)}
    if env.probe():
        reply["running"] = True
        reply["stills"] = env.summary()
    else:
        reply["starting"] = True  # models still loading
    return reply


def _stop(env) -> bool:
    record = projects._read("studio.json")
    if not record.get("pid") or not env.alive(record["pid"]):
        projects.clear_studio_record()
        return False
    env.kill(record["pid"])
    projects.clear_studio_record()
    return True


def _open(folder: Path, env) -> dict:
    if not projects.is_project(folder):
        return {"ok": False, "error": f"{folder} isn't a footsight project"}
    _stop(env)
    pid = env.launch(folder)
    projects.save_studio_record(pid, folder)
    projects.remember(folder)
    return {"ok": True, "path": str(folder), "starting": True}


def _new_project(request: dict, env) -> dict:
    name = (request.get("name") or "").strip() or "Match"
    location = request.get("location")
    if request.get("choose"):
        location = env.choose(str(projects.last_location() or Path.home()))
        if location is None:
            return {"ok": False, "cancelled": True}
    location = Path(location or projects.last_location() or Path.home())
    folder, existing = projects.project_folder_for(location, name)
    if not existing:
        try:
            (folder / "originals").mkdir(parents=True)
        except OSError as error:
            return {"ok": False, "error": f"Can't create a project there ({location}): {error.strerror or error}"}
    projects.set_last_location(location)
    reply = _open(folder, env)
    if reply["ok"]:
        reply = {"ok": True, "path": str(folder), "existing": existing, "starting": True}
    return reply


def _recent() -> dict:
    location = projects.last_location()
    return {
        "ok": True,
        "projects": [{"name": p.name, "path": str(p), "stills": projects.still_count(p)} for p in projects.recent()],
        "last_location": str(location) if location else None,
    }


def handle(request: dict, env) -> dict:
    try:
        command = request.get("cmd")
        if command == "status":
            return _status(env)
        if command == "new_project":
            return _new_project(request, env)
        if command == "open":
            return _open(Path(request.get("path", "")), env)
        if command == "recent":
            return _recent()
        if command == "stop":
            return {"ok": True, "stopped": _stop(env)}
        return {"ok": False, "error": f"Unknown request {command!r}"}
    except Exception as error:  # never leave the extension without an answer
        return {"ok": False, "error": str(error) or error.__class__.__name__}


def main() -> None:
    request = read_message(sys.stdin.buffer)
    if request is not None:
        write_message(sys.stdout.buffer, handle(request, MacEnv()))


if __name__ == "__main__":
    main()
