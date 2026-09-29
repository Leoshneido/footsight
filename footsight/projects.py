"""Project bookkeeping for the extension workflow: where projects are, which
were used recently, the last folder picked, and which studio is running.

State lives in ~/.footsight (FOOTSIGHT_HOME overrides it, e.g. in tests):
  projects.json  {"recent": [paths], "last_location": path}
  studio.json    {"pid": int, "project": path}   -- the running studio
"""
import json
import os
import re
from pathlib import Path

RECENT_LIMIT = 8


def state_dir() -> Path:
    return Path(os.environ.get("FOOTSIGHT_HOME") or Path.home() / ".footsight")


def _read(name: str) -> dict:
    path = state_dir() / name
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _write(name: str, data: dict) -> None:
    state_dir().mkdir(parents=True, exist_ok=True)
    (state_dir() / name).write_text(json.dumps(data, indent=1))


def is_project(folder) -> bool:
    """A footsight project folder: a studio session, with its originals."""
    return (Path(folder) / "originals").is_dir()


def project_folder_for(location, name: str) -> tuple[Path, bool]:
    """Where a new project called `name` goes in `location`, and whether it
    is an existing project of that name (then it's reopened). A folder of
    that name that isn't a project is never taken over: "<name> (2)"..."""
    safe = re.sub(r'[\\/:*?"<>|]+', "-", name).strip(" .") or "Match"
    base = Path(location) / safe
    if not base.exists() or is_project(base):
        return base, is_project(base)
    n = 2
    while (Path(location) / f"{safe} ({n})").exists():
        n += 1
    return Path(location) / f"{safe} ({n})", False


def remember(folder) -> None:
    folder = str(Path(folder))
    data = _read("projects.json")
    recent = [p for p in data.get("recent", []) if p != folder]
    data["recent"] = [folder] + recent
    _write("projects.json", data)


def recent(limit: int = RECENT_LIMIT) -> list[Path]:
    """Recent projects, newest first, skipping any moved or deleted."""
    return [Path(p) for p in _read("projects.json").get("recent", []) if is_project(p)][:limit]


def last_location() -> Path | None:
    location = _read("projects.json").get("last_location")
    return Path(location) if location else None


def set_last_location(location) -> None:
    data = _read("projects.json")
    data["last_location"] = str(Path(location))
    _write("projects.json", data)


def still_count(folder) -> int:
    return len(list((Path(folder) / "originals").glob("[0-9][0-9][0-9].png")))


def save_studio_record(pid: int, folder) -> None:
    _write("studio.json", {"pid": pid, "project": str(Path(folder))})


def clear_studio_record() -> None:
    try:
        (state_dir() / "studio.json").unlink()
    except FileNotFoundError:
        pass


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running_studio(alive=pid_alive, probe=None) -> dict | None:
    """The recorded studio, if its process is alive and it answers (probe);
    a stale record (crashed or stopped studio) is cleared."""
    record = _read("studio.json")
    if not record.get("pid"):
        return None
    if not alive(record["pid"]):
        clear_studio_record()
        return None
    if probe is not None and not probe():
        return None
    return {"pid": record["pid"], "project": record["project"]}
