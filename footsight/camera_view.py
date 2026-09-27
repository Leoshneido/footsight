"""The camera-angle view: the still redrawn from its own broadcast camera.

Nothing from the broadcast image is reused (see MEMORY.md, 2026-09-27):
the stands, ad boards, pitch and players are all drawn. The pitch comes
from the top-down renderer's grass and markings, warped into the camera's
perspective with the still's homography; players stand where they were
detected, posed like the real ones when their pose can be trusted.

Layers, back to front: crowd, ground, far-side ad boards, goals, then
shadows, players and ball sorted far to near.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from footsight.posed_figure import draw_posed_player
from footsight.render import (
    ICON_HEIGHT_UNITS,
    PLAYER_HEIGHT_M,
    draw_ball,
    draw_markings,
    draw_player_icon,
    grass_image,
)

DEFAULT_SCALE = 1.5
PITCH_LENGTH_M, PITCH_WIDTH_M = 105.0, 68.0

# Ground: grass continues past the lines up to (and just under) the boards.
APRON_M = 4.5
TOP_PX_PER_M = 36  # resolution of the top-down source that gets warped

# Ad boards stand BOARD_OFFSET_M outside the lines, only on sides facing
# away from the camera; the logo repeats every BOARD_SEGMENT_M.
BOARD_OFFSET_M = 4.0
BOARD_HEIGHT_M = 0.9
BOARD_SEGMENT_M = 10.0
BOARD_COLOR = (18, 28, 48)
BOARD_TILE_W = 2220
LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "logo" / "footsight_logo_board.png"

# Goals: real dimensions; the net runs GOAL_DEPTH_M back, sloping down to
# GOAL_BACK_HEIGHT_FRACTION of the crossbar height, with a GOAL_ROOF_DEPTH_M roof.
GOAL_WIDTH_M = 7.32
GOAL_HEIGHT_M = 2.44
GOAL_DEPTH_M = 2.0
GOAL_ROOF_DEPTH_M = 1.0
GOAL_BACK_HEIGHT_FRACTION = 0.45
POST_THICKNESS_M = 0.12
POST_COLOR = (250, 250, 250)
POST_OUTLINE_COLOR = (40, 40, 40)
NET_COLOR = (235, 235, 235, 150)

# A vertical meter on screen, when too few players are detected to measure
# it: this many times the ground scale along the touchline at that row.
FALLBACK_VERTICAL_RATIO = 2.0

# Crowd: dark stands speckled with seeded, slightly darkened colored dots,
# growing toward the bottom (nearer rows).
STAND_COLOR = (34, 40, 58)
CROWD_PALETTE = [(210, 60, 60), (240, 240, 240), (60, 90, 200), (230, 200, 60), (90, 90, 110), (160, 60, 150)]
CROWD_DARKEN = 0.8
CROWD_PX_PER_DOT = 160
CROWD_SEED = 11

# The standing figure is drawn a little taller than its detection box, which
# stops at the top of the head rather than including the hair.
ICON_BOX_FACTOR = 1.05
# Ball: larger than life (0.11 m) so it stays visible, never below a floor;
# 0.3 m read as a third of a player's height near the camera.
BALL_RADIUS_M = 0.16
MIN_BALL_RADIUS_PX = 4.0


def scaled_homography(homography: np.ndarray, scale: float) -> np.ndarray:
    """Image->pitch homography for the still enlarged by `scale`."""
    return homography @ np.linalg.inv(np.diag([scale, scale, 1.0]))


def _apply(matrix: np.ndarray, point) -> np.ndarray:
    v = matrix @ np.array([point[0], point[1], 1.0])
    return v[:2] / v[2]


def vertical_scale(boxes, homography: np.ndarray, image_w: float):
    """Pixels per vertical meter at an image row. The homography only knows
    the ground plane, so height is measured off the players themselves: a
    straight-line fit of box height against foot row, a player being
    PLAYER_HEIGHT_M tall. Too few players to fit falls back to a multiple
    of the ground scale along the touchline."""
    rows = np.array([box[3] for box in boxes], dtype=float)
    heights = np.array([box[3] - box[1] for box in boxes], dtype=float)
    if len(boxes) >= 2 and np.ptp(rows) > 1e-6:
        slope, intercept = np.polyfit(rows, heights, 1)
        return lambda row: max(slope * row + intercept, 1.0) / PLAYER_HEIGHT_M

    to_image = np.linalg.inv(homography)

    def fallback(row):
        ground = _apply(homography, (image_w / 2, row))
        step = _apply(to_image, ground + np.array([1.0, 0.0])) - _apply(to_image, ground)
        return FALLBACK_VERTICAL_RATIO * float(np.linalg.norm(step))

    return fallback


def _in_front(to_image: np.ndarray, pitch_point, reference_sign: float) -> bool:
    w = (to_image @ np.array([pitch_point[0], pitch_point[1], 1.0]))[2]
    return w != 0 and np.sign(w) == reference_sign


def far_board_segments(homography: np.ndarray, image_w: int, image_h: int):
    """Ground-line endpoints (image pixels) of the ad-board sections, on the
    sides of the pitch facing away from the camera -- where moving outward
    from the pitch moves up the image at all, which includes a goal line
    receding diagonally when the camera looks toward a goal."""
    to_image = np.linalg.inv(homography)
    reference = np.sign((to_image @ np.array([0.0, 0.0, 1.0]))[2])
    half_x = PITCH_LENGTH_M / 2 + BOARD_OFFSET_M
    half_y = PITCH_WIDTH_M / 2 + BOARD_OFFSET_M
    sides = [
        ((-half_x, -half_y), (half_x, -half_y), (0.0, -1.0)),
        ((-half_x, half_y), (half_x, half_y), (0.0, 1.0)),
        ((-half_x, -half_y), (-half_x, half_y), (-1.0, 0.0)),
        ((half_x, -half_y), (half_x, half_y), (1.0, 0.0)),
    ]
    segments = []
    for start, end, outward in sides:
        start, end, outward = np.array(start), np.array(end), np.array(outward)
        middle = (start + end) / 2
        if not (_in_front(to_image, middle, reference) and _in_front(to_image, middle + outward, reference)):
            continue
        dx, dy = _apply(to_image, middle + outward) - _apply(to_image, middle)
        if dy >= -1e-6:  # moving outward doesn't go up the image: faces the camera
            continue
        count = int(np.ceil(np.linalg.norm(end - start) / BOARD_SEGMENT_M))
        points = [start + (end - start) * k / count for k in range(count + 1)]
        for a, b in zip(points, points[1:]):
            if not (_in_front(to_image, a, reference) and _in_front(to_image, b, reference)):
                continue
            left, right = _apply(to_image, a), _apply(to_image, b)
            xs, ys = (left[0], right[0]), (left[1], right[1])
            if max(xs) < -0.1 * image_w or min(xs) > 1.1 * image_w or max(ys) < -image_h or min(ys) > 2 * image_h:
                continue
            segments.append((left, right))
    return segments


def goal_frames(homography: np.ndarray, image_w: int, image_h: int) -> list[dict]:
    """Ground points (image pixels) of each goal in view: front posts on the
    goal line, the back of the net GOAL_DEPTH_M behind it and the edge of
    the roof. Goals out of frame or beyond the horizon are left out."""
    to_image = np.linalg.inv(homography)
    reference = np.sign((to_image @ np.array([0.0, 0.0, 1.0]))[2])
    half = GOAL_WIDTH_M / 2
    goals = []
    for side in (-1, 1):
        line = side * PITCH_LENGTH_M / 2
        corners = {
            "front_left": (line, -half), "front_right": (line, half),
            "back_left": (line + side * GOAL_DEPTH_M, -half), "back_right": (line + side * GOAL_DEPTH_M, half),
            "roof_left": (line + side * GOAL_ROOF_DEPTH_M, -half), "roof_right": (line + side * GOAL_ROOF_DEPTH_M, half),
        }
        if not all(_in_front(to_image, point, reference) for point in corners.values()):
            continue
        goal = {name: _apply(to_image, point) for name, point in corners.items()}
        xs = [p[0] for p in goal.values()]
        ys = [p[1] for p in goal.values()]
        if max(xs) < -0.1 * image_w or min(xs) > 1.1 * image_w or min(ys) > 2 * image_h or max(ys) < -image_h:
            continue
        goals.append(goal)
    return goals


def _draw_goals(image: Image.Image, goals, vertical) -> None:
    """Net first (a see-through grid over back, sides and roof), then the
    white frame on top. Drawn before the players, so a goalkeeper on his
    line stands in front of it."""
    from PIL import ImageDraw

    def up(point, meters):
        return point - np.array([0.0, meters * vertical(point[1])])

    for goal in goals:
        fl, fr = goal["front_left"], goal["front_right"]
        bl, br = goal["back_left"], goal["back_right"]
        flt, frt = up(fl, GOAL_HEIGHT_M), up(fr, GOAL_HEIGHT_M)
        rlt, rrt = up(goal["roof_left"], GOAL_HEIGHT_M), up(goal["roof_right"], GOAL_HEIGHT_M)
        back_height = GOAL_HEIGHT_M * GOAL_BACK_HEIGHT_FRACTION
        blt, brt = up(bl, back_height), up(br, back_height)

        net = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(net)
        line = max(1, int(round(0.02 * vertical(fl[1]))))

        def mesh(bottom_a, bottom_b, top_b, top_a, count):
            for t in np.linspace(0, 1, count):
                draw.line([tuple(bottom_a + (bottom_b - bottom_a) * t), tuple(top_a + (top_b - top_a) * t)], fill=NET_COLOR, width=line)
                draw.line([tuple(bottom_a + (top_a - bottom_a) * t), tuple(bottom_b + (top_b - bottom_b) * t)], fill=NET_COLOR, width=line)

        mesh(bl, br, brt, blt, 14)  # back
        mesh(fl, bl, blt, flt, 8)  # sides
        mesh(fr, br, brt, frt, 8)
        mesh(flt, frt, rrt, rlt, 8)  # roof
        image.paste(net, (0, 0), net)

        frame = ImageDraw.Draw(image)
        post = max(2, int(round(POST_THICKNESS_M * vertical(fl[1]) * 1.4)))
        for a, b in ((fl, flt), (fr, frt), (flt, frt)):
            frame.line([tuple(a), tuple(b)], fill=POST_OUTLINE_COLOR, width=post + 2)
        for a, b in ((fl, flt), (fr, frt), (flt, frt)):
            frame.line([tuple(a), tuple(b)], fill=POST_COLOR, width=post)


def _ground(homography: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """The mown pitch warped into the camera's view, and its coverage mask
    (0-1), with anything beyond the horizon masked out."""
    scale = TOP_PX_PER_M
    margin = int(round(APRON_M * scale))
    top_w = int(round(PITCH_LENGTH_M * scale)) + 2 * margin
    top_h = int(round(PITCH_WIDTH_M * scale)) + 2 * margin
    top = grass_image(top_w, top_h, margin, scale)
    draw_markings(top, PITCH_LENGTH_M, PITCH_WIDTH_M, margin, scale)

    top_to_pitch = np.array([
        [1 / scale, 0, -margin / scale - PITCH_LENGTH_M / 2],
        [0, 1 / scale, -margin / scale - PITCH_WIDTH_M / 2],
        [0, 0, 1],
    ])
    top_to_image = np.linalg.inv(homography) @ top_to_pitch
    ground = cv2.warpPerspective(np.asarray(top), top_to_image, (width, height), flags=cv2.INTER_LINEAR)
    coverage = cv2.warpPerspective(np.full((top_h, top_w), 255, np.uint8), top_to_image, (width, height), flags=cv2.INTER_LINEAR)

    # Pixels on the far side of the horizon would map through the homography
    # to mirrored ground; keep only the camera's side.
    center = _apply(np.linalg.inv(homography), (0.0, 0.0))
    reference = np.sign(homography[2] @ np.array([center[0], center[1], 1.0]))
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    same_side = np.sign(homography[2, 0] * xs + homography[2, 1] * ys + homography[2, 2]) == reference
    mask = (coverage.astype(np.float32) / 255.0) * same_side
    return ground, mask


def _crowd(width: int, height: int, stand_bottom: int) -> Image.Image:
    image = Image.new("RGB", (width, height), STAND_COLOR)
    if stand_bottom <= 0:
        return image
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(CROWD_SEED)
    count = int(width * stand_bottom / CROWD_PX_PER_DOT)
    xs = rng.uniform(0, width, count)
    ys = rng.uniform(0, stand_bottom, count)
    picks = rng.integers(len(CROWD_PALETTE), size=count)
    for x, y, pick in zip(xs, ys, picks):
        r = height * (0.0012 + 0.0022 * y / stand_bottom)
        color = tuple(int(c * CROWD_DARKEN) for c in CROWD_PALETTE[pick])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    return image


def _board_tile() -> Image.Image:
    """One board section: navy, with the footsight logo centered."""
    tile_h = int(BOARD_TILE_W * BOARD_HEIGHT_M / BOARD_SEGMENT_M)
    tile = Image.new("RGBA", (BOARD_TILE_W, tile_h), (*BOARD_COLOR, 255))
    if LOGO_PATH.exists():
        logo = Image.open(LOGO_PATH).convert("RGBA")
        logo_h = int(tile_h * 0.9)
        logo = logo.resize((int(logo.width * logo_h / logo.height), logo_h), Image.LANCZOS)
        tile.paste(logo, ((BOARD_TILE_W - logo.width) // 2, (tile_h - logo_h) // 2), logo)
    return tile


def _draw_boards(image: Image.Image, segments, vertical) -> None:
    tile = _board_tile()
    source = np.float32([(0, tile.height), (tile.width, tile.height), (tile.width, 0), (0, 0)])
    tile_array = np.asarray(tile)
    for left, right in sorted(segments, key=lambda s: max(s[0][1], s[1][1])):  # far first
        left_top = left - np.array([0.0, BOARD_HEIGHT_M * vertical(left[1])])
        right_top = right - np.array([0.0, BOARD_HEIGHT_M * vertical(right[1])])
        quad = np.float32([left, right, right_top, left_top])
        x0, y0 = np.floor(quad.min(axis=0)).astype(int)
        x1, y1 = np.ceil(quad.max(axis=0)).astype(int)
        if x1 - x0 <= 1 or y1 - y0 <= 1 or x1 - x0 > 3 * image.width:
            continue
        matrix = cv2.getPerspectiveTransform(source, quad - np.float32([x0, y0]))
        warped = cv2.warpPerspective(tile_array, matrix, (x1 - x0, y1 - y0), flags=cv2.INTER_AREA)
        section = Image.fromarray(warped, "RGBA")
        image.paste(section, (int(x0), int(y0)), section)


def render_camera_view(
    image_path: str,
    homography: np.ndarray,
    people,
    ball_pixel,
    output_path: str,
    scale: float = DEFAULT_SCALE,
) -> None:
    """Write the camera-angle view.

    people: [(box, category, (shirt, shorts), pose_or_None)] in still pixels;
    a None pose is drawn as the standing figure. ball_pixel: where the ball
    touches the ground, in still pixels, or None.
    """
    still_w, still_h = Image.open(image_path).size
    width, height = int(round(still_w * scale)), int(round(still_h * scale))
    H = scaled_homography(homography, scale)

    scaled = []
    for box, category, (shirt, shorts), pose in people:
        box = tuple(v * scale for v in box)
        if pose is not None:
            pose = pose.copy()
            pose[:, :2] *= scale
        scaled.append((box, shirt, shorts, pose))
    vertical = vertical_scale([box for box, *_ in scaled], H, width)

    ground, mask = _ground(H, width, height)
    has_ground = mask > 0.5
    first_ground_row = np.where(has_ground.any(axis=0), has_ground.argmax(axis=0), height)
    stand_bottom = int(first_ground_row.max())
    canvas = np.asarray(_crowd(width, height, stand_bottom)).astype(np.float32)
    canvas = ground * mask[..., None] + canvas * (1 - mask[..., None])
    image = Image.fromarray(canvas.clip(0, 255).astype(np.uint8))

    _draw_boards(image, far_board_segments(H, width, height), vertical)
    _draw_goals(image, goal_frames(H, width, height), vertical)

    drawables = [(box[3], "person", item) for item in scaled for box in [item[0]]]
    if ball_pixel is not None:
        ball = (ball_pixel[0] * scale, ball_pixel[1] * scale)
        drawables.append((ball[1], "ball", ball))
    for _, kind, item in sorted(drawables, key=lambda d: d[0]):  # far (higher up) first
        if kind == "ball":
            radius = max(MIN_BALL_RADIUS_PX, BALL_RADIUS_M * vertical(item[1]))
            draw_ball(image, (item[0], item[1] - radius), radius)
            continue
        (x1, y1, x2, y2), shirt, shorts, pose = item
        if pose is not None:
            draw_posed_player(image, pose, y2 - y1, shirt, shorts)
        else:
            draw_player_icon(image, ((x1 + x2) / 2, y2), shirt, shorts, (y2 - y1) * ICON_BOX_FACTOR / ICON_HEIGHT_UNITS)

    image.save(output_path)
