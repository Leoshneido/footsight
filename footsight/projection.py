import numpy as np


def project_point(H: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    """Apply a 3x3 homography to a single (x, y) pixel point."""
    x, y = point
    homogeneous = np.array([x, y, 1.0])
    result = H @ homogeneous
    result = result / result[2]
    return float(result[0]), float(result[1])


def project_points(H: np.ndarray, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [project_point(H, p) for p in points]


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
