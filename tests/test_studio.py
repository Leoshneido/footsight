import http.client
import io
import json
import threading
import time

import pytest
from PIL import Image

from footsight import edit, studio as studio_module
from footsight.studio import Studio

EXTENSION = {"X-Footsight-Capture": "1", "Origin": "chrome-extension://abcdefghijklmnop", "Content-Type": "image/png"}


def _png(color=(10, 120, 40)):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(buffer, format="PNG")
    return buffer.getvalue()


class FakePipeline:
    """Stands in for the slow models: analyze records the still it was
    given, render writes the files the editor serves."""

    def __init__(self, fail=None):
        self.fail = fail
        self.analyzed, self.rendered = [], []

    def analyze(self, input_path):
        self.analyzed.append(input_path)
        if self.fail:
            raise RuntimeError(self.fail)
        return {"version": 1, "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "detections": [], "poses": {}, "ball_pixel": None}

    def render(self, input_path, analysis, output_path, corrections=None):
        self.rendered.append((input_path, output_path, corrections))
        stem = output_path[: -len(".png")]
        for suffix in ("_camera.png", "_camera_background.png", "_camera_figures.png"):
            open(stem + suffix, "wb").write(_png())
        open(stem + "_camera_scene.json", "w").write(json.dumps({"version": 1, "players": []}))


def _wait(predicate, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def fake():
    return FakePipeline()


@pytest.fixture
def running(tmp_path, fake):
    s = Studio(tmp_path / "session", fake.analyze, fake.render)
    httpd = edit.make_server(s.stills, [], port=0, studio=s)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield s, httpd
    httpd.shutdown()
    httpd.server_close()
    s.stop()


def _request(httpd, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection(*httpd.server_address[:2], timeout=5)
    conn.request(method, path, body=body, headers=headers or {})
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, data


def test_capture_is_saved_numbered_and_processed(running, fake):
    s, httpd = running
    meta = json.dumps({"title": "Barca v Feyenoord", "host": "example.org", "time": 1234.5})

    status, data = _request(httpd, "POST", "/api/capture", _png(), {**EXTENSION, "X-Footsight-Meta": meta})
    assert status == 200 and json.loads(data) == {"id": 0, "name": "001_camera"}
    _request(httpd, "POST", "/api/capture", _png((200, 0, 0)), EXTENSION)

    assert _wait(lambda: [st.status for st in s.stills] == ["ready", "ready"])
    originals = s.dir / "originals"
    assert (originals / "001.png").exists() and (originals / "002.png").exists()
    assert json.loads((originals / "001.json").read_text())["title"] == "Barca v Feyenoord"
    assert fake.analyzed == [str(originals / "001.png"), str(originals / "002.png")]
    assert fake.rendered[0][1] == str(s.dir / "001.png")
    assert (s.dir / "001_analysis.json").exists()


def test_capture_must_come_from_the_extension(running):
    _, httpd = running
    web_page = {**EXTENSION, "Origin": "https://evil.example"}
    no_header = {k: v for k, v in EXTENSION.items() if k != "X-Footsight-Capture"}

    assert _request(httpd, "POST", "/api/capture", _png(), web_page)[0] == 403
    assert _request(httpd, "POST", "/api/capture", _png(), no_header)[0] == 403
    assert _request(httpd, "POST", "/api/capture", b"not a png", EXTENSION)[0] == 400


def test_capture_refuses_an_oversized_frame(running, monkeypatch):
    _, httpd = running
    monkeypatch.setattr(edit, "MAX_EXPORT_BYTES", 10)
    assert _request(httpd, "POST", "/api/capture", _png(), EXTENSION)[0] == 413


def test_a_still_that_cannot_be_processed_is_marked_failed(tmp_path):
    fake = FakePipeline(fail="Could not calibrate pitch")
    s = Studio(tmp_path / "session", fake.analyze, fake.render)
    try:
        s.capture(_png(), {})
        assert _wait(lambda: s.stills[0].status == "failed")
        assert "Could not calibrate" in s.stills[0].error
        assert (s.dir / "originals" / "001.png").exists()  # the original is kept
    finally:
        s.stop()


def test_listing_reports_studio_status_and_versions(running):
    s, httpd = running
    s.capture(_png(), {})
    assert _wait(lambda: s.stills[0].status == "ready")

    listing = json.loads(_request(httpd, "GET", "/api/stills")[1])
    assert listing["studio"] is True
    assert listing["stills"][0]["status"] == "ready" and listing["stills"][0]["version"] == 1


def test_corrections_re_render_the_still(running, fake):
    s, httpd = running
    s.capture(_png(), {})
    assert _wait(lambda: s.stills[0].status == "ready")

    status, data = _request(httpd, "GET", "/api/stills/0/corrections")
    assert status == 200 and json.loads(data) == {"removed": [], "added": [], "ball": {"mode": "auto"}, "sides": {}}

    fix = {"removed": [3], "added": [{"feet": [20.0, 10.0], "category": "team_b"}], "ball": {"mode": "set", "pixel": [5, 6]}}
    status, data = _request(httpd, "POST", "/api/stills/0/corrections", json.dumps(fix), {"Content-Type": "application/json"})
    assert status == 200 and json.loads(data)["version"] == 2

    applied = fake.rendered[-1][2]
    assert applied["removed"] == [3]
    assert applied["added"] == [{"id": 1000, "feet": [20.0, 10.0], "category": "team_b"}]
    assert json.loads((s.dir / "001_corrections.json").read_text()) == applied
    assert s.stills[0].version == 2


def test_invalid_corrections_are_refused(running):
    s, httpd = running
    s.capture(_png(), {})
    assert _wait(lambda: s.stills[0].status == "ready")
    for bad in ('{"removed": ["x"]}', '{"added": [{"feet": [1], "category": "team_a"}]}',
                '{"added": [{"feet": [1, 2], "category": "coach"}]}', '{"ball": {"mode": "set"}}', "nope"):
        assert _request(httpd, "POST", "/api/stills/0/corrections", bad)[0] == 400


def test_events_announce_new_and_finished_stills(running):
    s, httpd = running
    conn = http.client.HTTPConnection(*httpd.server_address[:2], timeout=5)
    conn.request("GET", "/api/events")
    response = conn.getresponse()
    assert response.getheader("Content-Type").startswith("text/event-stream")

    s.capture(_png(), {})
    statuses = []
    while "ready" not in statuses:
        line = response.fp.readline().decode().strip()
        if line.startswith("data:"):
            statuses.append(json.loads(line[5:])["still"]["status"])
    conn.close()
    assert statuses[0] == "queued" and statuses[-1] == "ready"


def test_a_past_session_can_be_reopened(tmp_path):
    fake = FakePipeline()
    first = Studio(tmp_path / "session", fake.analyze, fake.render)
    first.capture(_png(), {})
    assert _wait(lambda: first.stills[0].status == "ready")
    first.stop()
    (tmp_path / "session" / "originals" / "002.png").write_bytes(_png())  # captured but never processed

    reopened = Studio.open(tmp_path / "session", fake.analyze, fake.render)
    try:
        assert [st.name for st in reopened.stills] == ["001_camera", "002_camera"]
        assert reopened.stills[0].status == "ready"
        assert _wait(lambda: reopened.stills[1].status == "ready")
        assert fake.analyzed.count(str(tmp_path / "session" / "originals" / "001.png")) == 1  # not redone
    finally:
        reopened.stop()


def test_session_folder_name_has_the_date_and_session_name(tmp_path):
    folder = studio_module.session_folder(tmp_path, "Barca v Feyenoord", now=__import__("datetime").datetime(2026, 9, 28, 20, 45))
    assert folder == tmp_path / "2026-09-28 20-45 Barca v Feyenoord"


def test_capture_page_details_survive_accents(running):
    """The extension percent-encodes the page details (header values must be
    ASCII), so a title like "Barça 2–0 Feyenoord" arrives intact."""
    from urllib.parse import quote

    s, httpd = running
    meta = quote(json.dumps({"title": "Barça 2–0 Feyenoord"}))
    assert _request(httpd, "POST", "/api/capture", _png(), {**EXTENSION, "X-Footsight-Meta": meta})[0] == 200

    assert json.loads((s.dir / "originals" / "001.json").read_text())["title"] == "Barça 2–0 Feyenoord"


def test_a_fix_that_fails_to_render_is_reported_as_a_server_error(running, fake):
    """A valid fix whose re-render crashes must not be blamed on the editor
    ("invalid corrections"): it's a 500, and the previous images stay."""
    s, httpd = running
    s.capture(_png(), {})
    assert _wait(lambda: s.stills[0].status == "ready")

    def broken(*args, **kwargs):
        raise ValueError("render blew up")

    s.render = broken
    status, data = _request(httpd, "POST", "/api/stills/0/corrections", json.dumps({"removed": [1]}))
    assert status == 500 and b"render blew up" in data
    assert s.stills[0].version == 1
    assert not (s.dir / "001_corrections.json").exists()


def test_side_choices_are_validated():
    from footsight.studio import validate_corrections

    assert validate_corrections({"sides": {"20": "team_b"}})["sides"] == {"20": "team_b"}
    with pytest.raises(ValueError):
        validate_corrections({"sides": {"20": "coach"}})
    with pytest.raises(ValueError):
        validate_corrections({"sides": {"x": "team_a"}})


def test_the_studio_counts_idle_time_from_the_last_activity(tmp_path, fake, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(studio_module.time, "monotonic", lambda: clock[0])
    s = Studio(tmp_path / "session", fake.analyze, fake.render)
    try:
        clock[0] += 600
        assert s.idle_seconds() == pytest.approx(600)
        s.touch()
        assert s.idle_seconds() == pytest.approx(0)
        s.capture(_png(), {})  # a capture is activity too
        clock[0] += 30
        assert s.idle_seconds() == pytest.approx(30)
    finally:
        s.stop()


def test_the_studio_stops_itself_only_when_idle_and_no_editor_is_open(tmp_path, fake, monkeypatch):
    """Started from the extension, the studio shuts down after a long idle
    spell so it doesn't hold the models in memory forever -- but never while
    an editor page is open (listening for events)."""
    clock = [0.0]
    monkeypatch.setattr(studio_module.time, "monotonic", lambda: clock[0])
    s = Studio(tmp_path / "session", fake.analyze, fake.render)
    try:
        assert not s.should_stop_for_idle(idle_minutes=120)
        clock[0] += 121 * 60
        assert s.should_stop_for_idle(idle_minutes=120)
        editor = s.subscribe()
        assert not s.should_stop_for_idle(idle_minutes=120)
        s.unsubscribe(editor)
        assert not s.should_stop_for_idle(idle_minutes=0)  # 0 = never
    finally:
        s.stop()


def test_every_request_to_the_server_counts_as_activity(running, monkeypatch):
    s, httpd = running
    clock = [5000.0]
    monkeypatch.setattr(studio_module.time, "monotonic", lambda: clock[0])
    s.touch()
    clock[0] += 900
    _request(httpd, "GET", "/api/stills")
    assert s.idle_seconds() == pytest.approx(0)
