import math

from PIL import Image, ImageDraw

PITCH_COLOR = (34, 139, 34)
LINE_COLOR = (255, 255, 255)
BALL_FILL_COLOR = (255, 255, 255)
BALL_OUTLINE_COLOR = (0, 0, 0)

TEAM_A_COLOR = (30, 100, 220)
TEAM_B_COLOR = (220, 20, 60)
REFEREE_COLOR = (255, 215, 0)
ASSISTANT_REFEREE_COLOR = (150, 150, 90)

CATEGORY_COLORS = {
    "team_a": TEAM_A_COLOR,
    "team_b": TEAM_B_COLOR,
    "referee": REFEREE_COLOR,
    "assistant_referee": ASSISTANT_REFEREE_COLOR,
}

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


def _draw_player_icon(draw, position_px, color, head_radius_px=4, body_half_width_px=6, body_height_px=9):
    px, py = position_px
    total_height = head_radius_px * 2 + body_height_px
    top_y = py - total_height / 2

    head_center_y = top_y + head_radius_px
    draw.ellipse(
        [px - head_radius_px, head_center_y - head_radius_px, px + head_radius_px, head_center_y + head_radius_px],
        fill=color,
    )

    body_top_y = top_y + head_radius_px * 2
    body_bottom_y = body_top_y + body_height_px
    taper_px = body_half_width_px / 3
    draw.polygon(
        [
            (px - body_half_width_px, body_top_y),
            (px + body_half_width_px, body_top_y),
            (px + body_half_width_px - taper_px, body_bottom_y),
            (px - body_half_width_px + taper_px, body_bottom_y),
        ],
        fill=color,
    )


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
) -> None:
    scale = (image_width_px - 2 * margin_px) / pitch_length
    image_height_px = int(pitch_width * scale) + 2 * margin_px

    image = Image.new("RGB", (image_width_px, image_height_px), PITCH_COLOR)
    draw = ImageDraw.Draw(image)

    _draw_pitch_markings(draw, pitch_length, pitch_width, image_width_px, margin_px, scale)

    for position, category in player_positions:
        px, py = pitch_to_image_coords(position, pitch_length, pitch_width, image_width_px, margin_px)
        _draw_player_icon(draw, (px, py), CATEGORY_COLORS[category], head_radius_px=dot_radius_px // 2)

    if ball_position is not None:
        px, py = pitch_to_image_coords(ball_position, pitch_length, pitch_width, image_width_px, margin_px)
        draw.ellipse(
            [px - ball_radius_px, py - ball_radius_px, px + ball_radius_px, py + ball_radius_px],
            fill=BALL_FILL_COLOR,
            outline=BALL_OUTLINE_COLOR,
            width=2,
        )

    image.save(output_path)
