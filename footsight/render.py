import colorsys
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# 4200 px wide gives ~37 px per meter: high enough definition that the
# figures (~76 px tall) show their detail.
DEFAULT_IMAGE_WIDTH_PX = 4200
DEFAULT_MARGIN_PX = 160

# Markings are sized in meters so they scale with the image, and drawn
# MARKINGS_SUPERSAMPLE times larger then shrunk, so edges are smooth.
LINE_WIDTH_M = 0.12
SPOT_RADIUS_M = 0.15
MARKINGS_SUPERSAMPLE = 3
# Larger than a real ball (0.11 m) so it stays visible.
BALL_RADIUS_M = 0.33

# Mown grass: alternating light and dark bands of equal width from goal line
# to goal line (continuing into the margin), with a seeded texture -- fine
# grain plus larger faint patches -- so every render of the same input is
# identical.
GRASS_LIGHT_COLOR = (51, 95, 37)
GRASS_DARK_COLOR = (42, 85, 30)
GRASS_BANDS = 20
GRASS_GRAIN_SIGMA = 1.2
GRASS_PATCH_CELL_M = 2.2
GRASS_PATCH_STRENGTH = 0.028
GRASS_SEED = 7
LINE_COLOR = (255, 255, 255)
BALL_FILL_COLOR = (255, 255, 255)
BALL_OUTLINE_COLOR = (0, 0, 0)

TEAM_A_COLOR = (30, 100, 220)
TEAM_B_COLOR = (220, 20, 60)
REFEREE_COLOR = (255, 215, 0)
ASSISTANT_REFEREE_COLOR = (150, 150, 90)
# Keepers wear a kit distinct from both teams; both keepers share one color
# here, since they stand at opposite ends and can't be confused for one
# another.
GOALKEEPER_COLOR = (0, 230, 120)

CATEGORY_COLORS = {
    "team_a": TEAM_A_COLOR,
    "team_b": TEAM_B_COLOR,
    "goalkeeper": GOALKEEPER_COLOR,
    "referee": REFEREE_COLOR,
    "assistant_referee": ASSISTANT_REFEREE_COLOR,
}

# Shorts for anything drawn from the fixed palette (keepers, officials,
# and teams whose kit colors could not be read).
DEFAULT_SHORTS_COLOR = (30, 30, 30)

# Broadcast lighting washes kits out (Bayern's red reads brownish), so
# detected kit colors are drawn with their saturation scaled up by this
# much. Hue and brightness are kept; greys and whites stay grey and white.
KIT_SATURATION_BOOST = 1.5

# Player icon: a small broadcast-style figure (see draw_player_icon),
# drawn in abstract units and sized so head-top to soles is a real player's
# height on the pitch.
SUPERSAMPLE = 4
PLAYER_HEIGHT_M = 1.8
# Drawn this much larger than life so the figures read on the pitch.
ICON_SIZE_FACTOR = 1.15
ICON_HEIGHT_UNITS = 22.5  # head top at -11, soles at +11.5
ICON_TILE_HALF_UNITS = 15.0  # room for the arms and the blurred shadow
# Where the soles are, in drawing units below the tile center (the socks
# run from 9 to 11.5 units down).
ICON_FEET_UNITS = 11.0
ICON_SKIN_COLOR = (205, 150, 115)
ICON_HAIR_COLOR = (40, 28, 20)
ICON_SOCK_COLOR = (250, 250, 250)
ICON_OUTLINE_COLOR = (0, 0, 0)
ICON_SHADOW_COLOR = (10, 40, 10)
ICON_SHADOW_ALPHA = 110

# Standard FIFA pitch marking dimensions, in meters.
CENTER_CIRCLE_RADIUS = 9.15
PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_WIDTH = 40.32
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_WIDTH = 18.32
PENALTY_SPOT_DISTANCE = 11.0
CORNER_ARC_RADIUS = 1.0
# Half-angle of the visible penalty arc: the portion of the center-circle-radius
# circle around the penalty spot that lies outside the penalty area.
PENALTY_ARC_HALF_ANGLE_DEG = math.degrees(
    math.acos((PENALTY_AREA_DEPTH - PENALTY_SPOT_DISTANCE) / CENTER_CIRCLE_RADIUS)
)


def pitch_to_image_coords(
    point: tuple[float, float],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    image_width_px: int = DEFAULT_IMAGE_WIDTH_PX,
    margin_px: int = DEFAULT_MARGIN_PX,
) -> tuple[float, float]:
    x, y = point
    scale = (image_width_px - 2 * margin_px) / pitch_length
    px = margin_px + (x + pitch_length / 2.0) * scale
    py = margin_px + (y + pitch_width / 2.0) * scale
    return px, py


def _draw_rect(draw, to_px, x_a, x_b, y_a, y_b, line_width_px=2):
    p1 = to_px((x_a, y_a))
    p2 = to_px((x_b, y_b))
    x1, x2 = sorted([p1[0], p2[0]])
    y1, y2 = sorted([p1[1], p2[1]])
    draw.rectangle([x1, y1, x2, y2], outline=LINE_COLOR, width=line_width_px)


def _draw_spot(draw, center_px, radius_px=3):
    x, y = center_px
    draw.ellipse([x - radius_px, y - radius_px, x + radius_px, y + radius_px], fill=LINE_COLOR)


def _draw_pitch_markings(
    draw, pitch_length, pitch_width, image_width_px, margin_px, scale, line_width_px=2, spot_radius_px=3
):
    def to_px(point):
        return pitch_to_image_coords(point, pitch_length, pitch_width, image_width_px, margin_px)

    _draw_rect(draw, to_px, -pitch_length / 2, pitch_length / 2, -pitch_width / 2, pitch_width / 2, line_width_px)
    draw.line([to_px((0.0, -pitch_width / 2)), to_px((0.0, pitch_width / 2))], fill=LINE_COLOR, width=line_width_px)

    center = to_px((0.0, 0.0))
    circle_r_px = CENTER_CIRCLE_RADIUS * scale
    draw.ellipse(
        [center[0] - circle_r_px, center[1] - circle_r_px, center[0] + circle_r_px, center[1] + circle_r_px],
        outline=LINE_COLOR, width=line_width_px,
    )
    _draw_spot(draw, center, spot_radius_px)

    for side in (-1, 1):
        goal_line_x = side * pitch_length / 2

        pen_x = goal_line_x - side * PENALTY_AREA_DEPTH
        _draw_rect(draw, to_px, goal_line_x, pen_x, -PENALTY_AREA_WIDTH / 2, PENALTY_AREA_WIDTH / 2, line_width_px)

        goal_area_x = goal_line_x - side * GOAL_AREA_DEPTH
        _draw_rect(draw, to_px, goal_line_x, goal_area_x, -GOAL_AREA_WIDTH / 2, GOAL_AREA_WIDTH / 2, line_width_px)

        spot_x = goal_line_x - side * PENALTY_SPOT_DISTANCE
        spot_px = to_px((spot_x, 0.0))
        _draw_spot(draw, spot_px, spot_radius_px)

        arc_r_px = CENTER_CIRCLE_RADIUS * scale
        if side == -1:
            start_angle, end_angle = -PENALTY_ARC_HALF_ANGLE_DEG, PENALTY_ARC_HALF_ANGLE_DEG
        else:
            start_angle, end_angle = 180 - PENALTY_ARC_HALF_ANGLE_DEG, 180 + PENALTY_ARC_HALF_ANGLE_DEG
        draw.arc(
            [spot_px[0] - arc_r_px, spot_px[1] - arc_r_px, spot_px[0] + arc_r_px, spot_px[1] + arc_r_px],
            start=start_angle, end=end_angle, fill=LINE_COLOR, width=line_width_px,
        )

        corner_r_px = CORNER_ARC_RADIUS * scale
        for corner_y, quadrant in ((-pitch_width / 2, "near"), (pitch_width / 2, "far")):
            corner_px = to_px((goal_line_x, corner_y))
            if side == -1 and quadrant == "near":
                start, end = 0, 90
            elif side == -1 and quadrant == "far":
                start, end = 270, 360
            elif side == 1 and quadrant == "near":
                start, end = 90, 180
            else:
                start, end = 180, 270
            draw.arc(
                [corner_px[0] - corner_r_px, corner_px[1] - corner_r_px,
                 corner_px[0] + corner_r_px, corner_px[1] + corner_r_px],
                start=start, end=end, fill=LINE_COLOR, width=line_width_px,
            )


def vivid_kit_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """A detected kit color with its saturation boosted for display."""
    hue, saturation, value = colorsys.rgb_to_hsv(*(channel / 255 for channel in color))
    red, green, blue = colorsys.hsv_to_rgb(hue, min(1.0, saturation * KIT_SATURATION_BOOST), value)
    return (int(round(red * 255)), int(round(green * 255)), int(round(blue * 255)))


def paste_over(target: Image.Image, tile: Image.Image, xy: tuple[int, int]) -> None:
    """Draw an RGBA tile over target at xy. On an RGB image this is a
    masked paste; on an RGBA layer (the camera view's transparent figures
    layer) it is a proper "over" composite, so soft edges keep their color
    and coverage instead of being blended with transparent black."""
    if target.mode != "RGBA":
        target.paste(tile, xy, tile)
        return
    x, y = xy
    left, top = max(0, x), max(0, y)
    right, bottom = min(target.width, x + tile.width), min(target.height, y + tile.height)
    if right <= left or bottom <= top:
        return
    target.alpha_composite(tile.crop((left - x, top - y, right - x, bottom - y)), (left, top))


def category_kit(category, kit_colors):
    """(shirt, shorts) to draw a category in: its detected kit, saturation
    boosted, or the fixed palette shirt with dark shorts."""
    if category in kit_colors:
        shirt, shorts = (vivid_kit_color(color) for color in kit_colors[category])
        return shirt, shorts
    return CATEGORY_COLORS[category], DEFAULT_SHORTS_COLOR


def draw_player_icon(image, position_px, shirt, shorts, unit_px):
    """A small player as seen from a high broadcast camera: soft shadow,
    legs with white socks, shorts, shirt with arms, head with hair.

    Drawn SUPERSAMPLE times larger on its own transparent tile and shrunk
    down (box filter, so flat areas keep their exact color), then pasted
    so the feet stand on position_px -- a player's position is their
    ground-contact point, and a ball at their feet in the still has to land
    at the icon's feet, not on its shirt. unit_px is the size of one
    drawing unit on the final image.
    """
    unit = unit_px * SUPERSAMPLE
    tile_half_px = math.ceil(ICON_TILE_HALF_UNITS * unit_px)
    tile_half = tile_half_px * SUPERSAMPLE
    tile_size = tile_half * 2
    tile = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    cx, cy = tile_half, tile_half

    def box(x_a, y_a, x_b, y_b):
        return [cx + x_a * unit, cy + y_a * unit, cx + x_b * unit, cy + y_b * unit]

    shadow = Image.new("L", tile.size, 0)
    ImageDraw.Draw(shadow).ellipse(box(-3.0, 7.0, 11.0, 12.0), fill=ICON_SHADOW_ALPHA)
    shadow = shadow.filter(ImageFilter.GaussianBlur(unit * 1.5))
    tile.paste((*ICON_SHADOW_COLOR, 255), (0, 0), shadow)

    draw = ImageDraw.Draw(tile)
    outline = max(1, int(unit * 0.4))
    for side in (-1, 1):
        draw.line(
            [(cx + side * 2.0 * unit, cy + 4.0 * unit), (cx + side * 2.5 * unit, cy + 10.0 * unit)],
            fill=ICON_SKIN_COLOR, width=int(unit * 2.2),
        )
        draw.ellipse(box(side * 2.5 - 1.6, 9.0, side * 2.5 + 1.6, 11.5), fill=ICON_SOCK_COLOR)
    draw.rounded_rectangle(box(-3.8, 1.0, 3.8, 5.0), radius=int(unit), fill=shorts, outline=ICON_OUTLINE_COLOR, width=outline)
    for side in (-1, 1):
        draw.line(
            [(cx + side * 4.0 * unit, cy - 4.5 * unit), (cx + side * 6.0 * unit, cy + 1.0 * unit)],
            fill=shirt, width=int(unit * 2.4),
        )
        draw.ellipse(box(side * 6.0 - 1.3, 0.2, side * 6.0 + 1.3, 2.6), fill=ICON_SKIN_COLOR)
    draw.rounded_rectangle(box(-4.5, -6.0, 4.5, 2.0), radius=int(unit * 2), fill=shirt, outline=ICON_OUTLINE_COLOR, width=outline)
    draw.ellipse(box(-2.6, -11.0, 2.6, -5.8), fill=ICON_SKIN_COLOR, outline=ICON_OUTLINE_COLOR, width=max(1, int(unit * 0.3)))
    draw.chord(box(-2.6, -11.0, 2.6, -5.8), 180, 360, fill=ICON_HAIR_COLOR)

    small = tile.resize((tile_size // SUPERSAMPLE, tile_size // SUPERSAMPLE), Image.BOX)
    px, py = position_px
    feet_above_center = int(round(ICON_FEET_UNITS * unit_px))
    paste_over(image, small, (int(round(px)) - tile_half_px, int(round(py)) - tile_half_px - feet_above_center))


def grass_image(image_width_px, image_height_px, margin_px, scale):
    """The mown, textured pitch surface as an RGB image."""
    columns = np.arange(image_width_px)
    band_width_px = (105.0 / GRASS_BANDS) * scale
    dark = (np.floor((columns - margin_px) / band_width_px).astype(int) % 2).astype(bool)
    shade = np.where(dark[:, None], GRASS_DARK_COLOR, GRASS_LIGHT_COLOR).astype(np.float32)
    grass = np.broadcast_to(shade[None, :, :], (image_height_px, image_width_px, 3)).copy()

    rng = np.random.default_rng(GRASS_SEED)
    grain = rng.normal(0.0, GRASS_GRAIN_SIGMA, (image_height_px, image_width_px)).astype(np.float32)
    cell_px = max(1, int(round(GRASS_PATCH_CELL_M * scale)))
    coarse = rng.normal(128.0, 40.0, (image_height_px // cell_px + 2, image_width_px // cell_px + 2))
    patches = np.asarray(
        Image.fromarray(coarse.clip(0, 255).astype(np.uint8)).resize((image_width_px, image_height_px), Image.BICUBIC),
        dtype=np.float32,
    ) - 128.0
    grass += (grain + patches * GRASS_PATCH_STRENGTH)[..., None]
    return Image.fromarray(grass.clip(0, 255).astype(np.uint8))


def draw_markings(image, pitch_length, pitch_width, margin_px, scale):
    """Paint the pitch markings onto image with anti-aliased edges: they are
    drawn as a mask MARKINGS_SUPERSAMPLE times larger, then shrunk."""
    factor = MARKINGS_SUPERSAMPLE
    mask = Image.new("L", (image.width * factor, image.height * factor), 0)
    draw = _MaskDraw(ImageDraw.Draw(mask))
    _draw_pitch_markings(
        draw, pitch_length, pitch_width, image.width * factor, margin_px * factor, scale * factor,
        line_width_px=max(1, int(round(LINE_WIDTH_M * scale * factor))),
        spot_radius_px=max(1, SPOT_RADIUS_M * scale * factor),
    )
    mask = mask.resize(image.size, Image.BOX)
    image.paste(LINE_COLOR, (0, 0), mask)


class _MaskDraw:
    """Lets the marking code, which draws in LINE_COLOR, draw onto a
    single-channel mask: any fill or outline becomes full coverage (255)."""

    def __init__(self, draw):
        self._draw = draw

    def __getattr__(self, name):
        method = getattr(self._draw, name)

        def draw_opaque(*args, **kwargs):
            for key in ("fill", "outline"):
                if kwargs.get(key) is not None:
                    kwargs[key] = 255
            return method(*args, **kwargs)

        return draw_opaque


def draw_ball(image, position_px, radius_px):
    """The ball, anti-aliased the same way as the player icons."""
    factor = SUPERSAMPLE
    half = math.ceil(radius_px) + 2
    tile = Image.new("RGBA", (half * 2 * factor, half * 2 * factor), (0, 0, 0, 0))
    center = half * factor
    big_radius = radius_px * factor
    ImageDraw.Draw(tile).ellipse(
        [center - big_radius, center - big_radius, center + big_radius, center + big_radius],
        fill=BALL_FILL_COLOR, outline=BALL_OUTLINE_COLOR, width=max(1, int(round(big_radius * 0.18))),
    )
    small = tile.resize((half * 2, half * 2), Image.BOX)
    px, py = position_px
    paste_over(image, small, (int(round(px)) - half, int(round(py)) - half))


def render_pitch(
    player_positions: list[tuple[tuple[float, float], str]],
    output_path: str,
    ball_position: tuple[float, float] | None = None,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    image_width_px: int = DEFAULT_IMAGE_WIDTH_PX,
    margin_px: int = DEFAULT_MARGIN_PX,
    player_height_m: float = PLAYER_HEIGHT_M,
    ball_radius_m: float = BALL_RADIUS_M,
    kit_colors: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] | None = None,
) -> None:
    """kit_colors maps a category to its detected (shirt, shorts) colors;
    any category not in it is drawn in its fixed CATEGORY_COLORS shirt."""
    kit_colors = kit_colors or {}
    scale = (image_width_px - 2 * margin_px) / pitch_length
    image_height_px = int(pitch_width * scale) + 2 * margin_px

    image = grass_image(image_width_px, image_height_px, margin_px, scale)
    draw_markings(image, pitch_length, pitch_width, margin_px, scale)

    icon_unit_px = player_height_m * ICON_SIZE_FACTOR * scale / ICON_HEIGHT_UNITS
    for position, category in player_positions:
        px, py = pitch_to_image_coords(position, pitch_length, pitch_width, image_width_px, margin_px)
        shirt, shorts = category_kit(category, kit_colors)
        draw_player_icon(image, (px, py), shirt, shorts, icon_unit_px)

    if ball_position is not None:
        px, py = pitch_to_image_coords(ball_position, pitch_length, pitch_width, image_width_px, margin_px)
        draw_ball(image, (px, py), ball_radius_m * scale)

    image.save(output_path)
