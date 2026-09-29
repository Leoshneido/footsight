import io
import json
import struct

import pytest

from footsight import host, projects


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("FOOTSIGHT_HOME", str(tmp_path / "state"))
    return tmp_path / "state"


class FakeEnv:
    """Stands in for the Mac side: the studio process, the Finder picker,
    and the studio's HTTP answer."""

    def __init__(self, picked=None):
        self.picked = picked
        self.launched, self.killed, self.pickers = [], [], []
        self.live = set()
        self.answering = True
        self.summary_reply = {"ready": 0, "processing": 0, "queued": 0, "failed": 0}
        self.next_pid = 100

    def launch(self, folder):
        self.next_pid += 1
        self.launched.append(folder)
        self.live.add(self.next_pid)
        return self.next_pid

    def choose(self, default):
        self.pickers.append(default)
        return self.picked

    def alive(self, pid):
        return pid in self.live

    def probe(self):
        return self.answering

    def kill(self, pid):
        self.killed.append(pid)
        self.live.discard(pid)

    def summary(self):
        return self.summary_reply


def _project(folder):
    (folder / "originals").mkdir(parents=True)
    return folder


def test_messages_use_chromes_length_prefixed_json():
    buffer = io.BytesIO()
    host.write_message(buffer, {"ok": True, "name": "Barça"})
    raw = buffer.getvalue()
    assert struct.unpack("<I", raw[:4])[0] == len(raw) - 4
    assert host.read_message(io.BytesIO(raw)) == {"ok": True, "name": "Barça"}


def test_status_when_nothing_is_running():
    reply = host.handle({"cmd": "status"}, FakeEnv())
    assert reply == {"ok": True, "running": False, "starting": False, "project": None, "stills": {}, "url": host.STUDIO_URL}


def test_new_project_in_a_chosen_folder_starts_the_studio(tmp_path):
    env = FakeEnv(picked=str(tmp_path))
    reply = host.handle({"cmd": "new_project", "name": "Barca v Feyenoord", "choose": True}, env)

    folder = tmp_path / "Barca v Feyenoord"
    assert reply == {"ok": True, "path": str(folder), "existing": False, "starting": True}
    assert projects.is_project(folder)
    assert env.launched == [folder]
    assert projects.last_location() == tmp_path
    assert projects.recent() == [folder]


def test_new_project_in_the_last_location_skips_the_picker(tmp_path):
    projects.set_last_location(tmp_path)
    env = FakeEnv()
    reply = host.handle({"cmd": "new_project", "name": "Match 2026-09-29", "choose": False}, env)

    assert reply["ok"] and reply["path"] == str(tmp_path / "Match 2026-09-29")
    assert env.pickers == []


def test_cancelling_the_folder_picker_does_nothing(tmp_path):
    env = FakeEnv(picked=None)
    reply = host.handle({"cmd": "new_project", "name": "x", "choose": True}, env)
    assert reply == {"ok": False, "cancelled": True}
    assert env.launched == []


def test_a_location_that_cannot_be_written_is_refused(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        reply = host.handle({"cmd": "new_project", "name": "x", "choose": False, "location": str(locked)}, FakeEnv())
        assert reply["ok"] is False and "Can't create a project there" in reply["error"]
    finally:
        locked.chmod(0o700)


def test_opening_a_project_stops_the_running_one_first(tmp_path):
    env = FakeEnv()
    first, second = _project(tmp_path / "a"), _project(tmp_path / "b")
    host.handle({"cmd": "open", "path": str(first)}, env)
    running_pid = env.next_pid

    reply = host.handle({"cmd": "open", "path": str(second)}, env)

    assert reply == {"ok": True, "path": str(second), "starting": True}
    assert env.killed == [running_pid]
    assert env.launched == [first, second]
    assert projects.recent()[0] == second


def test_opening_a_folder_that_isnt_a_project_is_refused(tmp_path):
    (tmp_path / "photos").mkdir()
    reply = host.handle({"cmd": "open", "path": str(tmp_path / "photos")}, FakeEnv())
    assert reply["ok"] is False


def test_status_reports_the_running_project_and_its_stills(tmp_path):
    env = FakeEnv()
    folder = _project(tmp_path / "Barca v Feyenoord")
    host.handle({"cmd": "open", "path": str(folder)}, env)
    env.summary_reply = {"ready": 7, "processing": 1, "queued": 0, "failed": 0}

    reply = host.handle({"cmd": "status"}, env)
    assert reply["running"] is True and reply["starting"] is False
    assert reply["project"] == {"name": "Barca v Feyenoord", "path": str(folder)}
    assert reply["stills"] == {"ready": 7, "processing": 1, "queued": 0, "failed": 0}


def test_status_while_the_models_are_still_loading(tmp_path):
    env = FakeEnv()
    folder = _project(tmp_path / "p")
    host.handle({"cmd": "open", "path": str(folder)}, env)
    env.answering = False

    reply = host.handle({"cmd": "status"}, env)
    assert reply["running"] is False and reply["starting"] is True and reply["project"]["path"] == str(folder)


def test_status_explains_a_studio_that_died_while_starting(tmp_path):
    env = FakeEnv()
    folder = _project(tmp_path / "p")
    host.handle({"cmd": "open", "path": str(folder)}, env)
    (folder / "footsight.log").write_text("Loading models…\nOSError: [Errno 48] Address already in use\n")
    env.live.clear()  # the studio exited

    reply = host.handle({"cmd": "status"}, env)
    assert reply["running"] is False and reply["starting"] is False
    assert "Address already in use" in reply["error"]


def test_recent_lists_projects_with_their_still_counts(tmp_path):
    env = FakeEnv()
    folder = _project(tmp_path / "Barca v Feyenoord")
    (folder / "originals" / "001.png").write_bytes(b"x")
    host.handle({"cmd": "open", "path": str(folder)}, env)
    projects.set_last_location(tmp_path)

    reply = host.handle({"cmd": "recent"}, env)
    assert reply == {
        "ok": True,
        "projects": [{"name": "Barca v Feyenoord", "path": str(folder), "stills": 1}],
        "last_location": str(tmp_path),
    }


def test_stop_ends_the_running_studio(tmp_path):
    env = FakeEnv()
    host.handle({"cmd": "open", "path": str(_project(tmp_path / "p"))}, env)
    pid = env.next_pid

    assert host.handle({"cmd": "stop"}, env) == {"ok": True, "stopped": True}
    assert env.killed == [pid]
    assert host.handle({"cmd": "stop"}, env) == {"ok": True, "stopped": False}


def test_an_unknown_request_is_an_error_not_a_crash():
    assert host.handle({"cmd": "format_disk"}, FakeEnv())["ok"] is False
