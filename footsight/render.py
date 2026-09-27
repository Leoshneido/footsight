import math

from PIL import Image, ImageDraw, ImageFilter

PITCH_COLOR = (34, 139, 34)
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

# Player icon: a small broadcast-style figure (see _draw_player_icon).
SUPERSAMPLE = 4
ICON_UNIT_PX = 1.8
ICON_TILE_HALF_PX = 26
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
    image_width_px: int = 1050,
    margin_px: int = 40,
) -> tuple[float, float]:
    x, y = point
    scale = (image_width_px - 2 * margin_px) / pitch_length
    px = margin_px + (x + pitch_length / 2.0) * scale
    py = margin_px + (y + pitch_width / 2.0) * scale
    return px, py


def _draw_rect(draw, to_px, x_a, x_b, y_a, y_b):
    p1 = to_px((x_a, y_a))
    p2 = to_px((x_b, y_b))
    x1, x2 = sorted([p1[0], p2[0]])
    y1, y2 = sorted([p1[1], p2[1]])
    draw.rectangle([x1, y1, x2, y2], outline=LINE_COLOR, width=2)


def _draw_spot(draw, center_px, radius_px=3):
    x, y = center_px
    draw.ellipse([x - radius_px, y - radius_px, x + radius_px, y + radius_px], fill=LINE_COLOR)


def _draw_pitch_markings(draw, pitch_length, pitch_width, image_width_px, margin_px, scale):
    def to_px(point):
        return pitch_to_image_coords(point, pitch_length, pitch_width, image_width_px, margin_px)

    _draw_rect(draw, to_px, -pitch_length / 2, pitch_length / 2, -pitch_width / 2, pitch_width / 2)
    draw.line([to_px((0.0, -pitch_width / 2)), to_px((0.0, pitch_width / 2))], fill=LINE_COLOR, width=2)

    center = to_px((0.0, 0.0))
    circle_r_px = CENTER_CIRCLE_RADIUS * scale
    draw.ellipse(
        [center[0] - circle_r_px, center[1] - circle_r_px, center[0] + circle_r_px, center[1] + circle_r_px],
        outline=LINE_COLOR, width=2,
    )
    _draw_spot(draw, center)

    for side in (-1, 1):
        goal_line_x = side * pitch_length / 2

        pen_x = goal_line_x - side * PENALTY_AREA_DEPTH
        _draw_rect(draw, to_px, goal_line_x, pen_x, -PENALTY_AREA_WIDTH / 2, PENALTY_AREA_WIDTH / 2)

        goal_area_x = goal_line_x - side * GOAL_AREA_DEPTH
        _draw_rect(draw, to_px, goal_line_x, goal_area_x, -GOAL_AREA_WIDTH / 2, GOAL_AREA_WIDTH / 2)

        spot_x = goal_line_x - side * PENALTY_SPOT_DISTANCE
        spot_px = to_px((spot_x, 0.0))
        _draw_spot(draw, spot_px)

        arc_r_px = CENTER_CIRCLE_RADIUS * scale
        if side == -1:
            start_angle, end_angle = -PENALTY_ARC_HALF_ANGLE_DEG, PENALTY_ARC_HALF_ANGLE_DEG
        else:
            start_angle, end_angle = 180 - PENALTY_ARC_HALF_ANGLE_DEG, 180 + PENALTY_ARC_HALF_ANGLE_DEG
        draw.arc(
            [spot_px[0] - arc_r_px, spot_px[1] - arc_r_px, spot_px[0] + arc_r_px, spot_px[1] + arc_r_px],
            start=start_angle, end=end_angle, fill=LINE_COLOR, width=2,
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
                start=start, end=end, fill=LINE_COLOR, width=2,
            )


def _draw_player_icon(image, position_px, shirt, shorts, size=1.0):
    """A small player as seen from a high broadcast camera: soft shadow,
    legs with white socks, shorts, shirt with arms, head with hair.

    Drawn SUPERSAMPLE times larger on its own transparent tile and shrunk
    down (box filter, so flat areas keep their exact color), then pasted
    centered on the shirt at position_px.
    """
    unit = ICON_UNIT_PX * size * SUPERSAMPLE
    tile_half = int(ICON_TILE_HALF_PX * size) * SUPERSAMPLE
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
    offset = int(ICON_TILE_HALF_PX * size)
    image.paste(small, (int(round(px)) - offset, int(round(py)) - offset), small)


def render_pitch(
    player_positions: list[tuple[tuple[float, float], str]],
    output_path: str,
    ball_position: tuple[float, float] | None = None,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    image_width_px: int = 1050,
    margin_px: int = 40,
    dot_radius_px: int = 8,
    ball_radius_px: int = 6,
    kit_colors: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] | None = None,
) -> None:
    """kit_colors maps a category to its detected (shirt, shorts) colors;
    any category not in it is drawn in its fixed CATEGORY_COLORS shirt."""
    kit_colors = kit_colors or {}
    scale = (image_width_px - 2 * margin_px) / pitch_length
    image_height_px = int(pitch_width * scale) + 2 * margin_px

    image = Image.new("RGB", (image_width_px, image_height_px), PITCH_COLOR)
    draw = ImageDraw.Draw(image)

    _draw_pitch_markings(draw, pitch_length, pitch_width, image_width_px, margin_px, scale)

    for position, category in player_positions:
        px, py = pitch_to_image_coords(position, pitch_length, pitch_width, image_width_px, margin_px)
        shirt, shorts = kit_colors.get(category, (CATEGORY_COLORS[category], DEFAULT_SHORTS_COLOR))
        _draw_player_icon(image, (px, py), shirt, shorts, size=dot_radius_px / 8)

    if ball_position is not None:
        px, py = pitch_to_image_coords(ball_position, pitch_length, pitch_width, image_width_px, margin_px)
        draw.ellipse(
            [px - ball_radius_px, py - ball_radius_px, px + ball_radius_px, py + ball_radius_px],
            fill=BALL_FILL_COLOR,
            outline=BALL_OUTLINE_COLOR,
            width=2,
        )

    image.save(output_path)
