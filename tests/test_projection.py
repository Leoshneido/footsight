import numpy as np
import pytest

from footsight.projection import (
    project_point,
    project_points,
    bbox_to_ground_point,
    filter_to_pitch,
)


def test_project_point_identity_homography():
    H = np.eye(3)
    assert project_point(H, (10.0, 20.0)) == (10.0, 20.0)


def test_project_point_scaling_homography():
    H = np.array([
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 1.0],
    ])
    result = project_point(H, (40.0, 60.0))
    assert result == pytest.approx((20.0, 30.0))


def test_project_points_applies_to_all():
    H = np.eye(3)
    points = [(1.0, 2.0), (3.0, 4.0)]
    assert project_points(H, points) == [(1.0, 2.0), (3.0, 4.0)]


def test_bbox_to_ground_point_uses_bottom_center():
    bbox = (10.0, 20.0, 30.0, 50.0)
    assert bbox_to_ground_point(bbox) == (20.0, 50.0)


def test_filter_to_pitch_keeps_points_within_margin():
    points = [(0.0, 0.0), (52.5, 34.0), (57.5, 39.0)]
    result = filter_to_pitch(points, pitch_length=105.0, pitch_width=68.0, margin=5.0)
    assert result == [(0.0, 0.0), (52.5, 34.0), (57.5, 39.0)]


def test_filter_to_pitch_discards_points_outside_margin():
    points = [(0.0, 0.0), (200.0, 0.0), (0.0, -100.0)]
    result = filter_to_pitch(points, pitch_length=105.0, pitch_width=68.0, margin=5.0)
    assert result == [(0.0, 0.0)]
