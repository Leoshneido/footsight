import pytest
from PIL import Image, ImageDraw

from footsight.ball_detection import find_ball, _player_search_window


def _make_pitch_with_player(path, player_box, ball_ellipse=None):
    image = Image.new("RGB", (200, 200), color=(34, 139, 34))  # green pitch
    draw = ImageDraw.Draw(image)
    draw.rectangle(list(player_box), fill=(0, 0, 200))  # jersey, not white/green
    if ball_ellipse is not None:
        draw.ellipse(list(ball_ellipse), fill=(255, 255, 255))
    image.save(path)


def test_player_search_window_pads_and_clamps():
    window = _player_search_window((50.0, 50.0, 70.0, 110.0), (200, 200), padding_ratio=0.5)
    assert window == (40.0, 20.0, 80.0, 140.0)


def test_player_search_window_clamps_to_image_bounds():
    window = _player_search_window((0.0, 0.0, 20.0, 20.0), (100, 100), padding_ratio=1.0)
    assert window == (0.0, 0.0, 40.0, 40.0)


def test_find_ball_locates_white_circle_near_player(tmp_path):
    image_path = tmp_path / "still.png"
    player_box = (50.0, 50.0, 70.0, 110.0)
    _make_pitch_with_player(image_path, player_box, ball_ellipse=(75, 100, 85, 110))

    box = find_ball(str(image_path), [player_box], padding_ratio=1.0)

    assert box is not None
    x1, y1, x2, y2 = box
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    assert center == pytest.approx((80.0, 105.0), abs=2.0)


def test_find_ball_separates_ball_touching_a_line(tmp_path):
    # A ball touching a thin white line merges into one elongated blob unless
    # the line is broken off first -- this is the real failure mode found on
    # actual broadcast footage (ball sitting on a pitch marking).
    image_path = tmp_path / "still.png"
    player_box = (50.0, 50.0, 70.0, 110.0)
    image = Image.new("RGB", (200, 200), color=(34, 139, 34))
    draw = ImageDraw.Draw(image)
    draw.rectangle(list(player_box), fill=(0, 0, 200))
    draw.line([(0, 105), (200, 105)], fill=(255, 255, 255), width=3)  # pitch line
    draw.ellipse([75, 100, 85, 110], fill=(255, 255, 255))  # ball touching the line
    image.save(image_path)

    box = find_ball(str(image_path), [player_box], padding_ratio=1.0)

    assert box is not None
    x1, y1, x2, y2 = box
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    assert center == pytest.approx((80.0, 105.0), abs=3.0)


def test_find_ball_returns_none_when_no_ball_present(tmp_path):
    image_path = tmp_path / "still.png"
    player_box = (50.0, 50.0, 70.0, 110.0)
    _make_pitch_with_player(image_path, player_box)

    box = find_ball(str(image_path), [player_box], padding_ratio=1.0)

    assert box is None


def test_find_ball_returns_none_when_no_players(tmp_path):
    image_path = tmp_path / "still.png"
    Image.new("RGB", (200, 200), color=(34, 139, 34)).save(image_path)

    box = find_ball(str(image_path), [])

    assert box is None
