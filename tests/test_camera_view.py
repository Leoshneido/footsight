import cv2
import numpy as np
import pytest
from PIL import Image

from footsight import camera_view
from footsight.camera_view import far_board_segments, render_camera_view, scaled_homography, vertical_scale

STILL_W, STILL_H = 600, 400
# A broadcast-like camera: far touchline across the top, near touchline
# beyond the bottom edge, goal lines converging upward.
PITCH_CORNERS = np.float32([(-52.5, -34), (52.5, -34), (52.5, 34), (-52.5, 34)])
IMAGE_CORNERS = np.float32([(60, 100), (540, 100), (700, 520), (-100, 520)])
PITCH_TO_IMAGE = cv2.getPerspectiveTransform(PITCH_CORNERS, IMAGE_CORNERS).astype(float)
H = np.linalg.inv(PITCH_TO_IMAGE)  # image -> pitch, like pitch_calibration returns


def _to_pitch(Hm, point):
    v = Hm @ np.array([point[0], point[1], 1.0])
    return v[:2] / v[2]


def _to_image(point):
    v = PITCH_TO_IMAGE @ np.array([point[0], point[1], 1.0])
    return v[:2] / v[2]


@pytest.fixture
def still(tmp_path):
    path = tmp_path / "still.png"
    Image.new("RGB", (STILL_W, STILL_H), (90, 140, 90)).save(path)
    return str(path)


@pytest.fixture(autouse=True)
def fast_ground(monkeypatch):
    """A coarser top-down source image keeps these renders quick."""
    monkeypatch.setattr(camera_view, "TOP_PX_PER_M", 8)


def test_scaled_homography_maps_scaled_pixels_to_the_same_pitch_point():
    Hs = scaled_homography(H, 1.5)
    assert _to_pitch(Hs, (450, 300)) == pytest.approx(_to_pitch(H, (300, 200)))


def test_vertical_scale_is_fitted_from_player_heights():
    """Players are ~1.8 m tall, so box height against foot row gives the
    on-screen size of a vertical metre at any row."""
    boxes = [(100, 810, 120, 900), (300, 540, 314, 600)]  # 90 px at row 900, 60 px at row 600
    scale_at = vertical_scale(boxes, H, STILL_W)

    assert scale_at(900) == pytest.approx(90 / 1.8)
    assert scale_at(750) == pytest.approx(75 / 1.8)


def test_vertical_scale_falls_back_to_the_ground_scale_without_enough_players():
    scale_at = vertical_scale([(100, 200, 120, 260)], H, STILL_W)

    row = 300
    ground = _to_pitch(H, (STILL_W / 2, row))
    step = _to_image(ground + np.array([1.0, 0.0])) - _to_image(ground)
    assert scale_at(row) == pytest.approx(2.0 * np.linalg.norm(step), rel=1e-6)


def test_far_board_segments_run_along_the_far_touchline_only():
    segments = far_board_segments(H, STILL_W, STILL_H)

    assert segments
    for left, right in segments:
        for end in (left, right):
            assert _to_pitch(H, end)[1] == pytest.approx(-38.0, abs=1e-6)


def _person(box, category="team_a", pose=None):
    return (box, category, ((245, 130, 30), (240, 240, 240)), pose)


def test_camera_view_is_one_and_a_half_times_the_still(still, tmp_path):
    out = tmp_path / "camera.png"
    render_camera_view(still, H, [], None, str(out))

    assert Image.open(out).size == (900, 600)


def test_camera_view_crowd_is_the_same_on_every_render(still, tmp_path):
    render_camera_view(still, H, [], None, str(tmp_path / "a.png"))
    render_camera_view(still, H, [], None, str(tmp_path / "b.png"))

    assert Image.open(tmp_path / "a.png").tobytes() == Image.open(tmp_path / "b.png").tobytes()


def test_camera_view_draws_players_after_the_boards(still, tmp_path, monkeypatch):
    """Players stand in front of the ad boards -- the linesman on the far
    touchline must not be hidden behind them."""
    calls = []
    real_boards, real_icon = camera_view._draw_boards, camera_view.draw_player_icon
    monkeypatch.setattr(camera_view, "_draw_boards", lambda *a, **k: (calls.append("boards"), real_boards(*a, **k)))
    monkeypatch.setattr(camera_view, "draw_player_icon", lambda *a, **k: (calls.append("player"), real_icon(*a, **k)))

    render_camera_view(still, H, [_person((290, 60, 310, 100))], None, str(tmp_path / "c.png"))

    assert calls == ["boards", "player"]


def test_camera_view_draws_the_standing_figure_when_there_is_no_pose(still, tmp_path, monkeypatch):
    drawn = []
    monkeypatch.setattr(camera_view, "draw_player_icon", lambda image, feet, shirt, shorts, unit: drawn.append(feet))
    monkeypatch.setattr(camera_view, "draw_posed_player", lambda *a: pytest.fail("no pose to draw"))

    render_camera_view(still, H, [_person((290, 200, 310, 260))], None, str(tmp_path / "c.png"))

    assert drawn == [pytest.approx((450.0, 390.0))]  # feet, in 1.5x pixels


def test_camera_view_poses_a_player_with_a_trusted_pose(still, tmp_path, monkeypatch):
    posed = []
    monkeypatch.setattr(camera_view, "draw_posed_player", lambda image, pose, height, shirt, shorts: posed.append((pose, height)))
    monkeypatch.setattr(camera_view, "draw_player_icon", lambda *a: pytest.fail("pose should be used"))
    pose = np.column_stack([np.full(17, 300.0), np.linspace(200, 260, 17), np.full(17, 0.9)])

    render_camera_view(still, H, [_person((290, 200, 310, 260), pose=pose)], None, str(tmp_path / "c.png"))

    [(drawn_pose, height)] = posed
    assert drawn_pose[:, :2] == pytest.approx(pose[:, :2] * 1.5)
    assert height == pytest.approx(60 * 1.5)


def test_camera_view_draws_the_ball_with_a_minimum_size(still, tmp_path, monkeypatch):
    balls = []
    monkeypatch.setattr(camera_view, "draw_ball", lambda image, center, radius: balls.append((center, radius)))

    render_camera_view(still, H, [], (300.0, 250.0), str(tmp_path / "c.png"))

    [(center, radius)] = balls
    assert radius >= camera_view.MIN_BALL_RADIUS_PX
    assert center[0] == pytest.approx(450.0)
    assert center[1] == pytest.approx(375.0 - radius)  # the ball sits on its ground point


def test_far_board_segments_include_a_goal_line_that_faces_away_from_the_camera():
    """Looking diagonally toward a goal (the Bayern still), the boards behind
    that goal recede up and to the side -- they face away from the camera
    too, and without them the crowd runs straight onto the grass."""
    corners = np.float32([(0, 150), (500, 80), (700, 300), (100, 500)])
    to_image = cv2.getPerspectiveTransform(PITCH_CORNERS, corners).astype(float)
    Hd = np.linalg.inv(to_image)

    def side(segment):
        (xa, ya), (xb, yb) = (np.round(_to_pitch(Hd, end), 3) for end in segment)
        if xa == xb:
            return ("x", float(xa))
        if ya == yb:
            return ("y", float(ya))
        return None

    sides = {side(segment) for segment in far_board_segments(Hd, 800, 600)}
    assert ("x", 56.5) in sides  # behind the right goal
    assert ("y", -38.0) in sides  # far touchline
    assert ("x", -56.5) not in sides and ("y", 38.0) not in sides  # the sides facing the camera


def test_camera_view_ball_is_in_proportion_to_the_players(still, tmp_path, monkeypatch):
    """Near the camera a 0.3 m ball read as a third of a player's height;
    it stays larger than life (0.11 m) but no more than ~1/5 of a player."""
    balls = []
    monkeypatch.setattr(camera_view, "draw_ball", lambda image, center, radius: balls.append(radius))
    boxes = [_person((100, 300, 120, 390)), _person((300, 140, 314, 200))]

    render_camera_view(still, H, boxes, (200.0, 390.0), str(tmp_path / "c.png"))

    player_height_at_ball = 90 * 1.5
    assert 2 * balls[0] <= player_height_at_ball / 5
