import json

import pytest

from footsight import projects


@pytest.fixture
def home(tmp_path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setenv("FOOTSIGHT_HOME", str(state))
    return state


def _make_project(folder):
    (folder / "originals").mkdir(parents=True)
    return folder


def test_state_lives_in_the_footsight_home(home):
    assert projects.state_dir() == home


def test_a_project_is_a_folder_with_originals(tmp_path):
    assert projects.is_project(_make_project(tmp_path / "a"))
    (tmp_path / "b").mkdir()
    assert not projects.is_project(tmp_path / "b")
    assert not projects.is_project(tmp_path / "missing")


def test_new_project_folder_is_location_plus_name(tmp_path):
    folder, existing = projects.project_folder_for(tmp_path, "Barca v Feyenoord")
    assert folder == tmp_path / "Barca v Feyenoord" and existing is False


def test_new_project_reopens_an_existing_project_of_that_name(tmp_path):
    _make_project(tmp_path / "Barca v Feyenoord")
    folder, existing = projects.project_folder_for(tmp_path, "Barca v Feyenoord")
    assert folder == tmp_path / "Barca v Feyenoord" and existing is True


def test_new_project_never_takes_over_a_folder_that_isnt_a_project(tmp_path):
    (tmp_path / "Holiday").mkdir()
    (tmp_path / "Holiday (2)").mkdir()
    folder, existing = projects.project_folder_for(tmp_path, "Holiday")
    assert folder == tmp_path / "Holiday (3)" and existing is False


def test_project_names_cant_escape_the_location(tmp_path):
    folder, _ = projects.project_folder_for(tmp_path, "../../etc/evil")
    assert folder.parent == tmp_path


def test_recent_projects_newest_first_and_capped(home, tmp_path):
    folders = [_make_project(tmp_path / f"p{i}") for i in range(10)]
    for folder in folders:
        projects.remember(folder)
    projects.remember(folders[3])  # reopened: moves to the top

    recent = projects.recent()
    assert recent[0] == folders[3]
    assert len(recent) == 8
    assert json.loads((home / "projects.json").read_text())["recent"][0] == str(folders[3])


def test_recent_skips_folders_that_were_moved_or_deleted(home, tmp_path):
    keep = _make_project(tmp_path / "keep")
    gone = _make_project(tmp_path / "gone")
    projects.remember(keep)
    projects.remember(gone)
    (gone / "originals").rmdir()
    gone.rmdir()

    assert projects.recent() == [keep]


def test_last_location_is_remembered(home, tmp_path):
    assert projects.last_location() is None
    projects.set_last_location(tmp_path)
    assert projects.last_location() == tmp_path


def test_still_counts_come_from_the_project_folder(tmp_path):
    folder = _make_project(tmp_path / "p")
    for name in ("001.png", "002.png", "003.png"):
        (folder / "originals" / name).write_bytes(b"x")
    assert projects.still_count(folder) == 3


def test_running_studio_needs_a_live_process_and_an_answer(home, tmp_path):
    folder = _make_project(tmp_path / "p")
    projects.save_studio_record(4242, folder)

    alive = lambda pid: pid == 4242
    assert projects.running_studio(alive=alive, probe=lambda: True) == {"pid": 4242, "project": str(folder)}
    assert projects.running_studio(alive=alive, probe=lambda: False) is None
    assert projects.running_studio(alive=lambda pid: False, probe=lambda: True) is None


def test_a_dead_studio_record_is_cleared(home, tmp_path):
    projects.save_studio_record(4242, tmp_path)
    projects.running_studio(alive=lambda pid: False, probe=lambda: False)
    assert not (home / "studio.json").exists()
