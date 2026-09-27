import pytest
from PIL import Image

from footsight.render import (
    pitch_to_image_coords,
    render_pitch,
    CATEGORY_COLORS,
    GOALKEEPER_COLOR,
    LINE_COLOR,
    CENTER_CIRCLE_RADIUS,
)


def test_pitch_to_image_coords_center_of_pitch():
    px, py = pitch_to_image_coords((0.0, 0.0), image_width_px=1050, margin_px=40)
    assert px == pytest.approx(525.0)


def test_pitch_to_image_coords_corner_maps_to_margin():
    px, py = pitch_to_image_coords((-52.5, -34.0), image_width_px=1050, margin_px=40)
    assert px == pytest.approx(40.0)
    assert py == pytest.approx(40.0)


def _pixel(image, point, **kwargs):
    px, py = pitch_to_image_coords(point, **kwargs)
    return image.getpixel((int(round(px)), int(round(py))))


def _any_pixel_near(image, point, color, radius=2, **kwargs):
    px, py = pitch_to_image_coords(point, **kwargs)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if image.getpixel((int(round(px)) + dx, int(round(py)) + dy)) == color:
                return True
    return False


def test_render_pitch_draws_team_a_head_at_expected_pixel(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch(
        [((0.0, 0.0), "team_a")], str(output_path),
        image_width_px=1050, margin_px=40,
    )

    image = Image.open(output_path)
    assert _any_pixel_near(image, (0.0, 0.0), (30, 100, 220), image_width_px=1050, margin_px=40)


def test_render_pitch_draws_team_b_and_referee_with_distinct_colors(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch(
        [((-30.0, 0.0), "team_b"), ((30.0, 0.0), "referee")], str(output_path),
        image_width_px=1050, margin_px=40,
    )

    image = Image.open(output_path)
    assert _any_pixel_near(image, (-30.0, 0.0), (220, 20, 60), image_width_px=1050, margin_px=40)
    assert _any_pixel_near(image, (30.0, 0.0), (255, 215, 0), image_width_px=1050, margin_px=40)


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


def test_render_pitch_draws_center_spot(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch([], str(output_path), image_width_px=1050, margin_px=40)

    image = Image.open(output_path)
    assert _pixel(image, (0.0, 0.0), image_width_px=1050, margin_px=40) == LINE_COLOR


def test_render_pitch_draws_center_circle_outline(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch([], str(output_path), image_width_px=1050, margin_px=40)

    image = Image.open(output_path)
    # A point directly to the right of center, at the circle's radius, should
    # land on the outline.
    assert _any_pixel_near(
        image, (CENTER_CIRCLE_RADIUS, 0.0), LINE_COLOR, radius=2, image_width_px=1050, margin_px=40
    )


def test_render_pitch_draws_penalty_spots_on_both_ends(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch([], str(output_path), image_width_px=1050, margin_px=40)

    image = Image.open(output_path)
    assert _pixel(image, (-52.5 + 11.0, 0.0), image_width_px=1050, margin_px=40) == LINE_COLOR
    assert _pixel(image, (52.5 - 11.0, 0.0), image_width_px=1050, margin_px=40) == LINE_COLOR


def test_render_pitch_draws_penalty_box_edges_on_both_ends(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch([], str(output_path), image_width_px=1050, margin_px=40)

    image = Image.open(output_path)
    # Front edge (inner edge, 16.5m from each goal line) of each penalty box,
    # at pitch-width center (y=0), should fall on the box outline.
    assert _any_pixel_near(
        image, (-52.5 + 16.5, 0.0), LINE_COLOR, radius=2, image_width_px=1050, margin_px=40
    )
    assert _any_pixel_near(
        image, (52.5 - 16.5, 0.0), LINE_COLOR, radius=2, image_width_px=1050, margin_px=40
    )


def test_render_pitch_draws_goalkeepers_in_their_own_color(tmp_path):
    """Goalkeepers wear a kit distinct from both teams and the officials, so
    the mockup gives them their own color rather than folding them into a
    team."""
    output_path = tmp_path / "mockup.png"
    render_pitch(
        [((-40.0, 0.0), "goalkeeper"), ((0.0, 0.0), "team_a"), ((10.0, 0.0), "team_b")],
        str(output_path),
        image_width_px=1050, margin_px=40,
    )

    image = Image.open(output_path)
    assert _any_pixel_near(image, (-40.0, 0.0), GOALKEEPER_COLOR, image_width_px=1050, margin_px=40)
    assert GOALKEEPER_COLOR not in (CATEGORY_COLORS["team_a"], CATEGORY_COLORS["team_b"], CATEGORY_COLORS["referee"])


def _pixels_near(image, point, radius, **kwargs):
    px, py = pitch_to_image_coords(point, **kwargs)
    return {
        image.getpixel((int(round(px)) + dx, int(round(py)) + dy))
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
    }


def test_render_pitch_draws_a_team_in_its_detected_kit(tmp_path):
    """Shirt and shorts both come from the still, so an orange team with
    white shorts is drawn that way instead of in the fixed team color."""
    output_path = tmp_path / "mockup.png"
    render_pitch(
        [((0.0, 0.0), "team_a")], str(output_path),
        image_width_px=1050, margin_px=40,
        kit_colors={"team_a": ((245, 130, 30), (240, 240, 240))},
    )

    near = _pixels_near(Image.open(output_path), (0.0, 0.0), radius=8, image_width_px=1050, margin_px=40)
    assert (245, 130, 30) in near
    assert (240, 240, 240) in near
    assert CATEGORY_COLORS["team_a"] not in near


def test_render_pitch_keeps_the_fixed_color_for_categories_without_a_detected_kit(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch(
        [((0.0, 0.0), "team_a"), ((20.0, 0.0), "goalkeeper")], str(output_path),
        image_width_px=1050, margin_px=40,
        kit_colors={"team_a": ((245, 130, 30), (240, 240, 240))},
    )

    near_keeper = _pixels_near(Image.open(output_path), (20.0, 0.0), radius=8, image_width_px=1050, margin_px=40)
    assert GOALKEEPER_COLOR in near_keeper


def test_render_pitch_player_icons_are_big_enough_to_read(tmp_path):
    """At the default 1050 px mockup width the figure has to be large enough
    to make out shirt and shorts -- ~25 px tall was too small to read."""
    output_path = tmp_path / "mockup.png"
    render_pitch([((0.0, 10.0), "team_a")], str(output_path), image_width_px=1050, margin_px=40)

    image = Image.open(output_path).convert("RGB")
    render_pitch([], str(tmp_path / "empty.png"), image_width_px=1050, margin_px=40)
    empty = Image.open(tmp_path / "empty.png").convert("RGB")
    from PIL import ImageChops
    top, bottom = ImageChops.difference(image, empty).getbbox()[1::2]
    assert bottom - top >= 36
