import pytest
from PIL import Image

from footsight.render import pitch_to_image_coords, render_pitch


def test_pitch_to_image_coords_center_of_pitch():
    px, py = pitch_to_image_coords((0.0, 0.0), image_width_px=1050, margin_px=40)
    assert px == pytest.approx(525.0)


def test_pitch_to_image_coords_corner_maps_to_margin():
    px, py = pitch_to_image_coords((-52.5, -34.0), image_width_px=1050, margin_px=40)
    assert px == pytest.approx(40.0)
    assert py == pytest.approx(40.0)


def test_render_pitch_draws_team_a_at_expected_pixel(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch(
        [((0.0, 0.0), "team_a")], str(output_path),
        image_width_px=1050, margin_px=40, dot_radius_px=8,
    )

    assert output_path.exists()
    image = Image.open(output_path)
    center_px, center_py = pitch_to_image_coords((0.0, 0.0), image_width_px=1050, margin_px=40)
    pixel = image.getpixel((int(center_px), int(center_py)))
    assert pixel == (30, 100, 220)


def test_render_pitch_draws_team_b_referee_with_distinct_colors(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch(
        [((-30.0, 0.0), "team_b"), ((30.0, 0.0), "referee")], str(output_path),
        image_width_px=1050, margin_px=40, dot_radius_px=8,
    )

    image = Image.open(output_path)
    team_b_px, team_b_py = pitch_to_image_coords((-30.0, 0.0), image_width_px=1050, margin_px=40)
    referee_px, referee_py = pitch_to_image_coords((30.0, 0.0), image_width_px=1050, margin_px=40)

    assert image.getpixel((int(team_b_px), int(team_b_py))) == (220, 20, 60)
    assert image.getpixel((int(referee_px), int(referee_py))) == (255, 215, 0)


def test_render_pitch_with_no_players_still_creates_pitch(tmp_path):
    output_path = tmp_path / "empty.png"
    render_pitch([], str(output_path))
    assert output_path.exists()


def test_render_pitch_draws_ball_at_expected_pixel(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch([], str(output_path), ball_position=(0.0, 0.0), image_width_px=1050, margin_px=40, ball_radius_px=6)

    image = Image.open(output_path)
    center_px, center_py = pitch_to_image_coords((0.0, 0.0), image_width_px=1050, margin_px=40)
    pixel = image.getpixel((int(center_px), int(center_py)))
    assert pixel == (255, 255, 255)


def test_render_pitch_with_no_ball_position_omits_ball(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch([], str(output_path), ball_position=None)
    assert output_path.exists()
