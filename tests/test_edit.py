import http.client
import io
import json
import threading

import pytest
from PIL import Image

from footsight import edit

SCENE = {"version": 1, "image": {"width": 8, "height": 6}, "players": [], "ball": None}


def _png(color=(10, 120, 40), size=(8, 6)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def folder(tmp_path):
    for name in ("a_camera", "b_camera"):
        (tmp_path / f"{name}.png").write_bytes(_png())
        (tmp_path / f"{name}_background.png").write_bytes(_png((1, 2, 3)))
        (tmp_path / f"{name}_scene.json").write_text(json.dumps(SCENE))
    (tmp_path / "a_camera_figures.png").write_bytes(_png((4, 5, 6)))
    # b_camera has no figures layer: generated before the editor existed
    return tmp_path


@pytest.fixture
def server(folder):
    stills, skipped = edit.find_stills(folder)
    httpd = edit.make_server(stills, skipped, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def _request(server, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection(*server.server_address[:2], timeout=5)
    conn.request(method, path, body=body, headers=headers or {})
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, response.getheader("Content-Type"), data


def test_find_stills_lists_complete_stills_and_skips_old_ones(folder):
    stills, skipped = edit.find_stills(folder)

    assert [s.name for s in stills] == ["a_camera"]
    assert skipped == ["b_camera"]


def test_find_stills_accepts_a_single_camera_view(folder):
    stills, _ = edit.find_stills(folder / "a_camera.png")

    assert [s.name for s in stills] == ["a_camera"]


def test_server_listens_on_this_machine_only(server):
    assert server.server_address[0] == "127.0.0.1"


def test_api_lists_stills_and_skipped(server):
    status, kind, data = _request(server, "GET", "/api/stills")

    assert status == 200 and kind.startswith("application/json")
    assert json.loads(data) == {"stills": [{"id": 0, "name": "a_camera", "has_overlays": False}], "skipped": ["b_camera"]}


def test_api_serves_scene_and_layers(server, folder):
    status, _, data = _request(server, "GET", "/api/stills/0/scene")
    assert status == 200 and json.loads(data) == SCENE

    status, kind, data = _request(server, "GET", "/api/stills/0/background")
    assert status == 200 and kind == "image/png" and data == (folder / "a_camera_background.png").read_bytes()

    status, _, data = _request(server, "GET", "/api/stills/0/figures")
    assert status == 200 and data == (folder / "a_camera_figures.png").read_bytes()


def test_overlays_are_only_there_once_saved(server, folder):
    assert _request(server, "GET", "/api/stills/0/overlays")[0] == 404

    saved = {"version": 1, "overlays": [{"type": "ring", "player": 0, "color": "yellow"}], "spotlight": False}
    status, _, _ = _request(server, "POST", "/api/stills/0/overlays", json.dumps(saved), {"Content-Type": "application/json"})
    assert status == 200
    assert json.loads((folder / "a_camera_overlays.json").read_text()) == saved

    status, _, data = _request(server, "GET", "/api/stills/0/overlays")
    assert status == 200 and json.loads(data) == saved
    assert json.loads(_request(server, "GET", "/api/stills")[2])["stills"][0]["has_overlays"] is True


def test_invalid_overlays_are_refused(server, folder):
    for body in ("not json", json.dumps({"version": 9, "overlays": []}), json.dumps({"version": 1, "overlays": [{"type": "laser"}]})):
        assert _request(server, "POST", "/api/stills/0/overlays", body)[0] == 400
    assert not (folder / "a_camera_overlays.json").exists()


def test_export_saves_the_png_next_to_the_still(server, folder):
    png = _png((200, 10, 10))
    status, _, _ = _request(server, "POST", "/api/stills/0/export", png, {"Content-Type": "image/png"})

    assert status == 200
    assert (folder / "a_camera_edited.png").read_bytes() == png


def test_export_refuses_anything_but_a_png(server, folder):
    assert _request(server, "POST", "/api/stills/0/export", b"GIF89a....")[0] == 400
    assert not (folder / "a_camera_edited.png").exists()


def test_export_refuses_an_oversized_upload(server, monkeypatch):
    monkeypatch.setattr(edit, "MAX_EXPORT_BYTES", 10)
    assert _request(server, "POST", "/api/stills/0/export", _png())[0] == 413


def test_unknown_still_and_paths_outside_the_editor_are_not_served(server):
    assert _request(server, "GET", "/api/stills/5/scene")[0] == 404
    assert _request(server, "GET", "/api/stills/0/secrets")[0] == 404
    assert _request(server, "GET", "/editor/../edit.py")[0] == 404
    assert _request(server, "GET", "/editor/%2e%2e/edit.py")[0] == 404


def test_editor_files_are_served(server):
    status, kind, data = _request(server, "GET", "/editor/geometry.js")

    assert status == 200 and "javascript" in kind and b"export function" in data


def test_the_editor_page_loads_the_editor_and_stacks_the_layers(server):
    """Ground graphics sit between the background and the figures, so they
    lie on the grass under the players; tags go on top."""
    status, kind, page = _request(server, "GET", "/")

    assert status == 200 and kind.startswith("text/html")
    page = page.decode()
    assert '<script type="module" src="/editor/editor.js">' in page
    order = [page.index(f'id="{layer}"') for layer in ("background", "ground", "preview", "figures", "veil", "tags")]
    assert order == sorted(order)


def test_the_toolbar_can_be_hidden_and_brought_back(server):
    """The toolbar sits on the right and stays on unless the user ticks
    "Hide toolbar"; a tab at the edge (or B) brings it back."""
    page = _request(server, "GET", "/")[2].decode()

    assert 'id="hide-toolbar" type="checkbox"' in page
    assert 'id="toolbar-tab"' in page
