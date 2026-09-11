import cv2
import numpy as np
from PIL import Image

# HSV thresholds for a white/light-colored ball against a green pitch and
# colored jerseys: low saturation, high brightness.
BALL_SATURATION_MAX = 60
BALL_VALUE_MIN = 170


def _player_search_window(
    player_box: tuple[float, float, float, float],
    image_size: tuple[int, int],
    padding_ratio: float = 0.5,
) -> tuple[float, float, float, float]:
    """Expand a player's box outward -- the ball, when at their feet, often
    sits just outside the box a person detector draws."""
    x1, y1, x2, y2 = player_box
    pad_x, pad_y = (x2 - x1) * padding_ratio, (y2 - y1) * padding_ratio
    image_width, image_height = image_size
    return (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(image_width), x2 + pad_x),
        min(float(image_height), y2 + pad_y),
    )


def _circularity(contour) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return 4 * np.pi * area / (perimeter ** 2)


def _find_ball_in_window(
    image_bgr: np.ndarray,
    window: tuple[float, float, float, float],
    min_radius_px: float,
    max_radius_px: float,
    min_circularity: float,
) -> tuple[tuple[float, float, float, float], float] | None:
    """Search one crop for a small, round, white-ish blob. Returns
    (bbox_in_crop_coordinates, circularity) for the best candidate that beats
    min_circularity, or None."""
    x1, y1, x2, y2 = (int(v) for v in window)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, BALL_VALUE_MIN), (179, BALL_SATURATION_MAX, 255))
    # Break thin connections (pitch lines, sock edges) from rounder blobs
    # before finding contours -- a ball touching a line otherwise merges
    # into one elongated shape with much lower circularity than the ball
    # alone would have.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = np.pi * min_radius_px ** 2
    max_area = np.pi * max_radius_px ** 2

    best = None
    best_circularity = min_circularity
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        circularity = _circularity(contour)
        if circularity >= best_circularity:
            best_circularity = circularity
            bx, by, bw, bh = cv2.boundingRect(contour)
            best = (float(bx), float(by), float(bx + bw), float(by + bh))

    if best is None:
        return None
    return best, best_circularity


def find_ball(
    image_path: str,
    player_boxes: list[tuple[float, float, float, float]],
    padding_ratio: float = 0.5,
    min_radius_px: float = 3.0,
    max_radius_px: float = 25.0,
    min_circularity: float = 0.65,
) -> tuple[float, float, float, float] | None:
    """Find the ball via classical color/shape segmentation near each
    detected player's feet, rather than the learned person/ball detector.

    The ball is often too small (~10-15px in a broadcast-wide still) for a
    general-purpose detector to find, and frequently sits right against a
    player's body -- exactly where a generic detector is least reliable. A
    small, round, white-ish blob against the green pitch and colored
    jerseys is something classical segmentation handles well.
    """
    if not player_boxes:
        return None

    image = Image.open(image_path).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    best_box = None
    best_circularity = min_circularity
    for player_box in player_boxes:
        window = _player_search_window(player_box, image.size, padding_ratio)
        offset_x, offset_y = window[0], window[1]

        result = _find_ball_in_window(image_bgr, window, min_radius_px, max_radius_px, best_circularity)
        if result is None:
            continue
        (bx1, by1, bx2, by2), circularity = result
        best_circularity = circularity
        best_box = (bx1 + offset_x, by1 + offset_y, bx2 + offset_x, by2 + offset_y)

    return best_box
