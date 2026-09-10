import numpy as np


def project_point(H: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    """Apply a 3x3 homography to a single (x, y) pixel point."""
    x, y = point
    homogeneous = np.array([x, y, 1.0])
    result = H @ homogeneous
    result = result / result[2]
    return float(result[0]), float(result[1])


def project_points(H: np.ndarray, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Project each point, discarding any on the wrong side of the homography's horizon.

    The pitch center (world (0,0)) is always a valid on-pitch point by
    construction. Its corresponding image pixel is used as a reference: any
    input point whose homogeneous sign disagrees with the reference's would
    otherwise divide through to a mirrored-but-plausible-looking position
    (e.g. a spectator detected above the pitch's vanishing line).
    """
    center_pixel = np.linalg.inv(H) @ np.array([0.0, 0.0, 1.0])
    ref_x, ref_y = center_pixel[0] / center_pixel[2], center_pixel[1] / center_pixel[2]
    ref_w = H[2, 0] * ref_x + H[2, 1] * ref_y + H[2, 2]

    result = []
    for x, y in points:
        w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        if w == 0 or (w > 0) != (ref_w > 0):
            continue
        result.append(project_point(H, (x, y)))
    return result


def bbox_to_ground_point(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Bottom-center of a bounding box (x1, y1, x2, y2) -- a player's ground-contact point."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def filter_to_pitch(
    points: list[tuple[float, float]],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    margin: float = 5.0,
) -> list[tuple[float, float]]:
    """Discard points that fall well outside the real pitch boundary (pitch centered at origin)."""
    half_length = pitch_length / 2.0 + margin
    half_width = pitch_width / 2.0 + margin
    return [
        (x, y) for x, y in points
        if -half_length <= x <= half_length and -half_width <= y <= half_width
    ]
