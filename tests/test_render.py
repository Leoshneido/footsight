import pytest
from PIL import Image

from footsight.render import (
    pitch_to_image_coords,
    render_pitch,
    CATEGORY_COLORS,
    GOALKEEPER_COLOR,
    LINE_COLOR,
    CENTER_CIRCLE_RADIUS,
    vivid_kit_color,
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


def _icon_pixels(image, point, **kwargs):
    """Every color in the area a player figure occupies: it stands on its
    ground point, so the figure is above the point, not centered on it."""
    px, py = (int(round(v)) for v in pitch_to_image_coords(point, **kwargs))
    return {image.getpixel((x, y)) for x in range(px - 12, px + 13) for y in range(py - 40, py + 1)}


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
    assert (30, 100, 220) in _icon_pixels(image, (0.0, 0.0), image_width_px=1050, margin_px=40)


def test_render_pitch_draws_team_b_and_referee_with_distinct_colors(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch(
        [((-30.0, 0.0), "team_b"), ((30.0, 0.0), "referee")], str(output_path),
        image_width_px=1050, margin_px=40,
    )

    image = Image.open(output_path)
    assert (220, 20, 60) in _icon_pixels(image, (-30.0, 0.0), image_width_px=1050, margin_px=40)
    assert (255, 215, 0) in _icon_pixels(image, (30.0, 0.0), image_width_px=1050, margin_px=40)


def test_render_pitch_with_no_players_still_creates_pitch(tmp_path):
    output_path = tmp_path / "empty.png"
    render_pitch([], str(output_path))
    assert output_path.exists()


def test_render_pitch_draws_ball_at_expected_pixel(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch([], str(output_path), ball_position=(0.0, 0.0))

    image = Image.open(output_path).convert("RGB")
    center_px, center_py = pitch_to_image_coords((0.0, 0.0))
    pixel = image.getpixel((int(center_px), int(center_py)))
    assert pixel == (255, 255, 255)


def test_render_pitch_sizes_the_ball_in_meters(tmp_path):
    """The ball keeps the same size relative to the pitch at any width
    (radius ~0.33 m, larger than life so it stays visible)."""
    for image_width_px, margin_px in ((2100, 80), (4200, 160)):
        output_path = tmp_path / f"ball_{image_width_px}.png"
        render_pitch([], str(output_path), ball_position=(0.0, 20.0), image_width_px=image_width_px, margin_px=margin_px)
        image = Image.open(output_path).convert("RGB")
        px, py = pitch_to_image_coords((0.0, 20.0), image_width_px=image_width_px, margin_px=margin_px)
        pixels_per_meter = (image_width_px - 2 * margin_px) / 105.0
        white_run = sum(
            1 for dx in range(-40, 41)
            if image.getpixel((int(px) + dx, int(py))) == (255, 255, 255)
        )
        assert 0.4 <= white_run / pixels_per_meter <= 0.7, (image_width_px, white_run)


def test_render_pitch_with_no_ball_position_omits_ball(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch([], str(output_path), ball_position=None)
    assert output_path.exists()


def test_render_pitch_draws_center_spot(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch([], str(output_path))

    image = Image.open(output_path)
    assert _pixel(image, (0.0, 0.0)) == LINE_COLOR


def test_render_pitch_draws_center_circle_outline(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch([], str(output_path))

    image = Image.open(output_path)
    # A point directly to the right of center, at the circle's radius, should
    # land on the outline.
    assert _any_pixel_near(
        image, (CENTER_CIRCLE_RADIUS, 0.0), LINE_COLOR, radius=2
    )


def test_render_pitch_draws_penalty_spots_on_both_ends(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch([], str(output_path))

    image = Image.open(output_path)
    assert _pixel(image, (-52.5 + 11.0, 0.0)) == LINE_COLOR
    assert _pixel(image, (52.5 - 11.0, 0.0)) == LINE_COLOR


def test_render_pitch_draws_penalty_box_edges_on_both_ends(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch([], str(output_path))

    image = Image.open(output_path)
    # Front edge (inner edge, 16.5m from each goal line) of each penalty box,
    # at pitch-width center (y=0), should fall on the box outline.
    assert _any_pixel_near(
        image, (-52.5 + 16.5, 0.0), LINE_COLOR, radius=2
    )
    assert _any_pixel_near(
        image, (52.5 - 16.5, 0.0), LINE_COLOR, radius=2
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
    assert GOALKEEPER_COLOR in _icon_pixels(image, (-40.0, 0.0), image_width_px=1050, margin_px=40)
    assert GOALKEEPER_COLOR not in (CATEGORY_COLORS["team_a"], CATEGORY_COLORS["team_b"], CATEGORY_COLORS["referee"])



def test_render_pitch_draws_a_team_in_its_detected_kit(tmp_path):
    """Shirt and shorts both come from the still, so an orange team with
    white shorts is drawn that way instead of in the fixed team color."""
    output_path = tmp_path / "mockup.png"
    render_pitch(
        [((0.0, 0.0), "team_a")], str(output_path),
        image_width_px=1050, margin_px=40,
        kit_colors={"team_a": ((245, 130, 30), (240, 240, 240))},
    )

    near = _icon_pixels(Image.open(output_path), (0.0, 0.0), image_width_px=1050, margin_px=40)
    assert vivid_kit_color((245, 130, 30)) in near
    assert (240, 240, 240) in near  # white has no saturation to boost
    assert CATEGORY_COLORS["team_a"] not in near


def test_render_pitch_keeps_the_fixed_color_for_categories_without_a_detected_kit(tmp_path):
    output_path = tmp_path / "mockup.png"
    render_pitch(
        [((0.0, 0.0), "team_a"), ((20.0, 0.0), "goalkeeper")], str(output_path),
        image_width_px=1050, margin_px=40,
        kit_colors={"team_a": ((245, 130, 30), (240, 240, 240))},
    )

    near_keeper = _icon_pixels(Image.open(output_path), (20.0, 0.0), image_width_px=1050, margin_px=40)
    assert GOALKEEPER_COLOR in near_keeper


def _figure_height_px(tmp_path, image_width_px, margin_px):
    from PIL import ImageChops

    render_pitch([((0.0, 10.0), "team_a")], str(tmp_path / "one.png"), image_width_px=image_width_px, margin_px=margin_px)
    render_pitch([], str(tmp_path / "none.png"), image_width_px=image_width_px, margin_px=margin_px)
    one = Image.open(tmp_path / "one.png").convert("RGB")
    none = Image.open(tmp_path / "none.png").convert("RGB")
    _, top, _, bottom = ImageChops.difference(one, none).getbbox()
    return bottom - top


def test_render_pitch_draws_players_a_little_larger_than_life(tmp_path):
    """A figure is sized relative to the pitch, whatever the image width: a
    real player's 1.8 m, drawn 15% larger so it reads (~2.07 m) -- at 4.3 m
    the first icons looked oversized. The measured height includes the soft
    shadow below the feet, hence the allowance above 2.07 m."""
    for image_width_px, margin_px in ((2100, 80), (4200, 160)):
        pixels_per_meter = (image_width_px - 2 * margin_px) / 105.0
        height_m = _figure_height_px(tmp_path, image_width_px, margin_px) / pixels_per_meter
        assert 2.35 <= height_m <= 2.9, (image_width_px, height_m)


def test_render_pitch_defaults_to_a_high_definition_width(tmp_path):
    """4200 px (~37 px per meter): at 2100 px the figures were too small to
    show much detail."""
    output_path = tmp_path / "mockup.png"
    render_pitch([], str(output_path))

    assert Image.open(output_path).width == 4200


def test_render_pitch_draws_detected_kits_more_vivid_than_the_dull_broadcast_color(tmp_path):
    """Broadcast lighting washes kits out -- Bayern's red reads (156, 62, 45),
    brownish. The drawn shirt keeps the hue but is more saturated."""
    import colorsys

    output_path = tmp_path / "mockup.png"
    dull_red = (156, 62, 45)
    render_pitch(
        [((0.0, 0.0), "team_a")], str(output_path),
        image_width_px=1050, margin_px=40,
        kit_colors={"team_a": (dull_red, (240, 240, 240))},
    )

    shirt = vivid_kit_color(dull_red)
    assert shirt in _icon_pixels(Image.open(output_path).convert("RGB"), (0.0, 0.0), image_width_px=1050, margin_px=40)
    dull_h, _, dull_s = colorsys.rgb_to_hls(*(c / 255 for c in dull_red))
    drawn_h, _, drawn_s = colorsys.rgb_to_hls(*(c / 255 for c in shirt))
    assert drawn_s > dull_s + 0.1, shirt
    assert abs(drawn_h - dull_h) < 0.02, shirt


def test_render_pitch_stands_each_figure_on_its_ground_point(tmp_path):
    """A player's position is where their feet touch the pitch, so the
    figure is drawn standing on it: shirt above the point, nothing of the
    figure below it (bar the shadow). That is what puts a ball at a
    player's feet in the still at the icon's feet on the mockup."""
    output_path = tmp_path / "mockup.png"
    kit = {"team_a": ((245, 130, 30), (240, 240, 240))}
    render_pitch([((0.0, 0.0), "team_a")], str(output_path), image_width_px=1050, margin_px=40, kit_colors=kit)

    image = Image.open(output_path).convert("RGB")
    px, py = (int(round(v)) for v in pitch_to_image_coords((0.0, 0.0), image_width_px=1050, margin_px=40))
    shirt = vivid_kit_color((245, 130, 30))
    rows_with_shirt = [
        y for y in range(py - 40, py + 40)
        if any(image.getpixel((x, y)) == shirt for x in range(px - 15, px + 16))
    ]
    assert rows_with_shirt, "shirt not drawn"
    assert max(rows_with_shirt) < py - 5, f"shirt reaches down to y={max(rows_with_shirt)}, feet point is y={py}"



def _band_mean(image, x_pitch_m, **kwargs):
    """Average color of a small patch of grass at a given distance along the
    pitch (on the y=28 m line, clear of the markings)."""
    import numpy as np

    px, py = (int(round(v)) for v in pitch_to_image_coords((x_pitch_m, 28.0), **kwargs))
    patch = np.array(image)[py - 5:py + 6, px - 5:px + 6].reshape(-1, 3)
    return patch.mean(axis=0), patch.std(axis=0)


def test_render_pitch_mows_the_grass_in_alternating_light_and_dark_bands(tmp_path):
    """20 bands of 5.25 m from goal line to goal line, alternating shades --
    the flat single green didn't look like a real pitch."""
    output_path = tmp_path / "mockup.png"
    render_pitch([], str(output_path), image_width_px=2100, margin_px=80)
    image = Image.open(output_path).convert("RGB")

    band_width_m = 105.0 / 20
    first_band_center = -52.5 + band_width_m / 2
    shades = [
        _band_mean(image, first_band_center + band * band_width_m, image_width_px=2100, margin_px=80)[0].sum()
        for band in (2, 3, 4, 5)
    ]
    assert shades[0] - shades[1] > 15 and shades[2] - shades[3] > 15, shades
    # the texture varies each band a little, but never enough to blur the pattern
    assert min(shades[0], shades[2]) > max(shades[1], shades[3]), shades


def test_render_pitch_gives_the_grass_a_subtle_texture(tmp_path):
    """Some grain so it doesn't read as flat paint, but faint -- the first,
    stronger texture was too busy."""
    output_path = tmp_path / "mockup.png"
    render_pitch([], str(output_path), image_width_px=2100, margin_px=80)
    image = Image.open(output_path).convert("RGB")

    _, spread = _band_mean(image, -30.0, image_width_px=2100, margin_px=80)
    assert 0.5 < spread.max() < 2.5, spread


def test_render_pitch_grass_is_a_deep_green(tmp_path):
    """Even the light bands are a deep green: the first two pitch colors
    read too bright."""
    output_path = tmp_path / "mockup.png"
    render_pitch([], str(output_path), image_width_px=2100, margin_px=80)
    image = Image.open(output_path).convert("RGB")

    band_width_m = 105.0 / 20
    light_band_center = -52.5 + band_width_m / 2
    shade, _ = _band_mean(image, light_band_center, image_width_px=2100, margin_px=80)
    assert shade.sum() < 200, shade
    assert shade[1] > shade[0] and shade[1] > shade[2], shade  # still green


def test_render_pitch_grass_is_the_same_on_every_render(tmp_path):
    """The texture is random-looking but seeded, so the same input always
    gives the same mockup."""
    render_pitch([], str(tmp_path / "a.png"), image_width_px=1050, margin_px=40)
    render_pitch([], str(tmp_path / "b.png"), image_width_px=1050, margin_px=40)

    assert Image.open(tmp_path / "a.png").tobytes() == Image.open(tmp_path / "b.png").tobytes()



def test_render_pitch_draws_lines_at_real_width(tmp_path):
    """Pitch lines are 12 cm wide, so they thicken with the image instead of
    staying a fixed 2 px that looks spidery at high definition."""
    output_path = tmp_path / "mockup.png"
    render_pitch([], str(output_path))
    image = Image.open(output_path).convert("RGB")

    px, py = (int(round(v)) for v in pitch_to_image_coords((0.0, 25.0)))
    bright = [dx for dx in range(-15, 16) if sum(image.getpixel((px + dx, py))) > 600]
    assert 3 <= len(bright) <= 6, bright


def test_render_pitch_draws_smooth_line_edges(tmp_path):
    """Lines and circles are anti-aliased: along the center circle there are
    in-between shades blending white into the grass, not a hard stair-step."""
    output_path = tmp_path / "mockup.png"
    render_pitch([], str(output_path))
    image = Image.open(output_path).convert("RGB")

    center_x, center_y = pitch_to_image_coords((0.0, 0.0))
    radius_px = CENTER_CIRCLE_RADIUS * (4200 - 2 * 160) / 105.0
    import math
    blended = 0
    for degrees in range(20, 70, 2):
        angle = math.radians(degrees)
        for offset in range(-6, 7):
            x = int(round(center_x + (radius_px + offset) * math.cos(angle)))
            y = int(round(center_y + (radius_px + offset) * math.sin(angle)))
            if 350 < sum(image.getpixel((x, y))) < 650:
                blended += 1
    assert blended >= 20, blended
