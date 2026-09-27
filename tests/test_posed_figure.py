import numpy as np
from PIL import Image

from footsight.posed_figure import draw_posed_player
from footsight.render import ICON_SKIN_COLOR

SHIRT, SHORTS = (245, 130, 30), (240, 240, 240)
HEIGHT = 200.0

# COCO order, in pixels on a 400x400 canvas: a player standing, arms bent.
STANDING = np.array([
    (200, 110), (196, 106), (204, 106), (190, 110), (210, 110),
    (180, 140), (220, 140), (170, 180), (230, 180), (175, 215), (225, 215),
    (186, 210), (214, 210), (184, 255), (216, 255), (184, 300), (216, 300),
], dtype=float)


def _render(points):
    pose = np.column_stack([points, np.full(17, 0.9)])
    image = Image.new("RGB", (400, 400), (40, 110, 40))
    draw_posed_player(image, pose, HEIGHT, SHIRT, SHORTS)
    return image


def _mid(a, b):
    return (int(round((a[0] + b[0]) / 2)), int(round((a[1] + b[1]) / 2)))


def test_posed_player_wears_the_shirt_between_shoulders_and_hips():
    image = _render(STANDING)
    shoulders = _mid(STANDING[5], STANDING[6])
    hips = _mid(STANDING[11], STANDING[12])

    assert image.getpixel(_mid(shoulders, hips)) == SHIRT


def test_posed_player_has_bare_forearms():
    image = _render(STANDING)

    assert image.getpixel(_mid(STANDING[7], STANDING[9])) == ICON_SKIN_COLOR
    assert image.getpixel(_mid(STANDING[8], STANDING[10])) == ICON_SKIN_COLOR


def test_side_on_player_keeps_a_body_instead_of_a_sliver():
    """Seen side-on, both shoulders (and hips) line up; the torso still gets
    a minimum width rather than collapsing to a line."""
    side = STANDING.copy()
    side[[5, 6], 0] = 200
    side[[11, 12], 0] = 200
    side[[7, 8, 9, 10]] = [(150, 170), (250, 170), (130, 200), (270, 200)]  # arms clear of the torso sample
    image = _render(side)
    torso_y = int((side[5, 1] + side[11, 1]) / 2)
    offset = int(0.08 * HEIGHT)

    assert image.getpixel((200 - offset, torso_y)) == SHIRT
    assert image.getpixel((200 + offset, torso_y)) == SHIRT


def test_posed_player_draws_shorts_below_the_hips():
    image = _render(STANDING)
    hip = STANDING[11]
    knee = STANDING[13]
    upper_thigh = (int(round(hip[0] + (knee[0] - hip[0]) * 0.25)), int(round(hip[1] + (knee[1] - hip[1]) * 0.25)))

    assert image.getpixel(upper_thigh) == SHORTS
