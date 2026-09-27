"""A cartoon footballer drawn on a real player's skeleton, so the figure
copies their pose (stride, arms, lean) instead of standing to attention.

Sizes are fractions of the player's on-screen height H. Drawn SUPERSAMPLE
times larger on its own tile and box-filtered down, like the standing icon,
so flat areas keep their exact colors.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from footsight.render import (
    ICON_HAIR_COLOR,
    ICON_OUTLINE_COLOR,
    ICON_SHADOW_ALPHA,
    ICON_SHADOW_COLOR,
    ICON_SKIN_COLOR,
    ICON_SOCK_COLOR,
    SUPERSAMPLE,
    paste_over,
)

JOINT_CONFIDENCE = 0.3
BOOT_COLOR = (25, 25, 25)

HEAD_RADIUS = 0.075
TORSO_MIN_WIDTH = 0.2  # side-on players keep a body
THIGH_WIDTH = 0.10
CALF_WIDTH = 0.075
UPPER_ARM_WIDTH = 0.062
FOREARM_WIDTH = 0.048
OUTLINE_WIDTH = 0.012
SHORTS_DOWN_THIGH = 0.5  # shorts cover the upper half of the thigh
SOCKS_FROM = 0.4  # socks cover the calf from 40% of the way down


def draw_posed_player(image: Image.Image, pose: np.ndarray, height_px: float, shirt, shorts) -> None:
    """Paint a posed figure onto image. pose is (17, 3): joint x, y in image
    pixels and confidence, COCO keypoint order."""
    confident = pose[:, 2] > JOINT_CONFIDENCE
    xs, ys = pose[confident, 0], pose[confident, 1]
    pad = height_px * 0.4
    left, top = int(xs.min() - pad), int(ys.min() - pad)
    right, bottom = int(xs.max() + pad), int(ys.max() + pad)
    tile_w, tile_h = right - left, bottom - top

    factor = SUPERSAMPLE
    unit = height_px * factor
    tile = Image.new("RGBA", (tile_w * factor, tile_h * factor), (0, 0, 0, 0))

    def at(i):
        return np.array([(pose[i, 0] - left) * factor, (pose[i, 1] - top) * factor])

    def ok(*joints):
        return all(confident[j] for j in joints)

    _draw_shadow(tile, pose, confident, left, top, factor, unit)
    draw = ImageDraw.Draw(tile)
    outline = unit * OUTLINE_WIDTH

    def limb(a, b, width_a, width_b, color):
        _tapered(draw, a, b, width_a + outline * 2, width_b + outline * 2, ICON_OUTLINE_COLOR)
        _tapered(draw, a, b, width_a, width_b, color)

    # legs: thigh, calf, sock, boot -- then shorts over the upper thigh
    for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
        if ok(hip, knee):
            limb(at(hip), at(knee), unit * THIGH_WIDTH, unit * CALF_WIDTH * 1.1, ICON_SKIN_COLOR)
        if ok(knee, ankle):
            sock_top = at(knee) + (at(ankle) - at(knee)) * SOCKS_FROM
            limb(at(knee), sock_top, unit * CALF_WIDTH * 1.1, unit * CALF_WIDTH, ICON_SKIN_COLOR)
            _tapered(draw, sock_top, at(ankle), unit * CALF_WIDTH, unit * CALF_WIDTH * 0.8, ICON_SOCK_COLOR)
            foot = at(ankle)
            r = unit * CALF_WIDTH * 0.7
            draw.ellipse([foot[0] - r, foot[1] - r * 0.6, foot[0] + r, foot[1] + r * 0.7], fill=BOOT_COLOR)
    for hip, knee in ((11, 13), (12, 14)):
        if ok(hip, knee):
            shorts_end = at(hip) + (at(knee) - at(hip)) * SHORTS_DOWN_THIGH
            _tapered(draw, at(hip), shorts_end, unit * THIGH_WIDTH * 1.25, unit * THIGH_WIDTH * 1.15, shorts)

    if ok(5, 6, 11, 12):
        _draw_torso(draw, at(5), at(6), at(11), at(12), unit, shirt, shorts, outline)

    # arms: shirt sleeve, then bare forearm and hand
    for shoulder, elbow, wrist in ((5, 7, 9), (6, 8, 10)):
        if ok(shoulder, elbow):
            limb(at(shoulder), at(elbow), unit * UPPER_ARM_WIDTH, unit * UPPER_ARM_WIDTH * 0.9, shirt)
        if ok(elbow, wrist):
            limb(at(elbow), at(wrist), unit * FOREARM_WIDTH, unit * FOREARM_WIDTH * 0.85, ICON_SKIN_COLOR)

    _draw_head(draw, pose, confident, at, unit, outline)

    small = tile.resize((tile_w, tile_h), Image.BOX)
    paste_over(image, small, (left, top))


def _tapered(draw, a, b, width_a, width_b, color):
    """A limb segment: a quad narrowing from width_a to width_b, with round ends."""
    direction = b - a
    length = np.linalg.norm(direction)
    if length < 1e-6:
        r = width_a / 2
        draw.ellipse([a[0] - r, a[1] - r, a[0] + r, a[1] + r], fill=color)
        return
    normal = np.array([-direction[1], direction[0]]) / length
    quad = [a + normal * width_a / 2, b + normal * width_b / 2, b - normal * width_b / 2, a - normal * width_a / 2]
    draw.polygon([tuple(p) for p in quad], fill=color)
    for point, width in ((a, width_a), (b, width_b)):
        r = width / 2
        draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r], fill=color)


def _draw_torso(draw, left_shoulder, right_shoulder, left_hip, right_hip, unit, shirt, shorts, outline):
    shoulder_mid = (left_shoulder + right_shoulder) / 2
    hip_mid = (left_hip + right_hip) / 2
    spine = hip_mid - shoulder_mid
    length = np.linalg.norm(spine) or 1.0
    across = np.array([-spine[1], spine[0]]) / length  # perpendicular to the spine

    min_width = unit * TORSO_MIN_WIDTH
    shoulder_width = max(np.linalg.norm(right_shoulder - left_shoulder) * 1.15, min_width)
    hip_width = max(np.linalg.norm(right_hip - left_hip) * 1.1, min_width * 0.9)
    chest = shoulder_mid + spine * 0.35
    chest_width = max(shoulder_width, hip_width) * 1.05

    outline_points = [
        shoulder_mid + across * shoulder_width / 2,
        chest + across * chest_width / 2,
        hip_mid + across * hip_width / 2,
        hip_mid - across * hip_width / 2,
        chest - across * chest_width / 2,
        shoulder_mid - across * shoulder_width / 2,
    ]
    draw.polygon([tuple(p) for p in outline_points], fill=shirt, outline=ICON_OUTLINE_COLOR, width=max(1, int(outline)))
    # waistband: the top of the shorts showing under the shirt hem
    band = unit * 0.035
    down = spine / length
    draw.polygon(
        [tuple(p) for p in (hip_mid + across * hip_width / 2, hip_mid - across * hip_width / 2,
                            hip_mid - across * hip_width / 2 + down * band, hip_mid + across * hip_width / 2 + down * band)],
        fill=shorts,
    )


def _draw_head(draw, pose, confident, at, unit, outline):
    face = [i for i in range(5) if confident[i]]
    if face:
        center = np.mean([at(i) for i in face], axis=0)
    elif confident[5] and confident[6]:
        center = (at(5) + at(6)) / 2 - np.array([0, unit * 0.1])
    else:
        return
    r = unit * HEAD_RADIUS
    box = [center[0] - r, center[1] - r, center[0] + r, center[1] + r]
    draw.ellipse(box, fill=ICON_SKIN_COLOR, outline=ICON_OUTLINE_COLOR, width=max(1, int(outline)))
    draw.chord(box, 180, 360, fill=ICON_HAIR_COLOR)


def _draw_shadow(tile, pose, confident, left, top, factor, unit):
    feet = [i for i in (15, 16) if confident[i]]
    if not feet:
        return
    fx = np.mean([pose[i, 0] for i in feet])
    fy = max(pose[i, 1] for i in feet)
    cx, cy = (fx - left) * factor + unit * 0.04, (fy - top) * factor
    shadow = Image.new("L", tile.size, 0)
    ImageDraw.Draw(shadow).ellipse([cx - unit * 0.18, cy - unit * 0.035, cx + unit * 0.22, cy + unit * 0.045], fill=ICON_SHADOW_ALPHA)
    shadow = shadow.filter(ImageFilter.GaussianBlur(unit * 0.02))
    tile.paste((*ICON_SHADOW_COLOR, 255), (0, 0), shadow)
