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


def test_render_pitch_creates_file_with_dot_at_expected_pixel(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch([(0.0, 0.0)], str(output_path), image_width_px=1050, margin_px=40, dot_radius_px=8)

    assert output_path.exists()
    image = Image.open(output_path)
    center_px, center_py = pitch_to_image_coords((0.0, 0.0), image_width_px=1050, margin_px=40)
    pixel = image.getpixel((int(center_px), int(center_py)))
    assert pixel == (220, 20, 60)


def test_render_pitch_with_no_players_still_creates_pitch(tmp_path):
    output_path = tmp_path / "empty.png"
    render_pitch([], str(output_path))
    assert output_path.exists()
