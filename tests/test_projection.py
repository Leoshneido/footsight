import numpy as np
import pytest

from footsight.projection import (
    project_point,
    project_points,
    bbox_to_ground_point,
    bbox_to_center_point,
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


def test_bbox_to_center_point_uses_box_center():
    bbox = (10.0, 20.0, 30.0, 60.0)
    assert bbox_to_center_point(bbox) == (20.0, 40.0)


def test_filter_to_pitch_keeps_points_within_margin():
    points = [(0.0, 0.0), (52.5, 34.0), (57.5, 39.0)]
    result = filter_to_pitch(points, pitch_length=105.0, pitch_width=68.0, margin=5.0)
    assert result == [(0.0, 0.0), (52.5, 34.0), (57.5, 39.0)]


def test_filter_to_pitch_discards_points_outside_margin():
    points = [(0.0, 0.0), (200.0, 0.0), (0.0, -100.0)]
    result = filter_to_pitch(points, pitch_length=105.0, pitch_width=68.0, margin=5.0)
    assert result == [(0.0, 0.0)]


def test_project_points_filters_points_across_the_horizon():
    # Bottom row [0, 1, -50] gives w = y - 50: a horizon at y=50.
    # The pitch-center world point (0,0) maps back to image pixel (0,0),
    # whose w is -50 (negative) -- so points with w<0 (y<50) are kept,
    # and points with w>0 (y>=50) are on the wrong side and dropped.
    H = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, -50.0],
    ])
    points = [(10.0, 0.0), (10.0, 100.0)]
    result = project_points(H, points)
    assert len(result) == 1
    assert result[0] == pytest.approx((-0.2, 0.0))


def test_project_points_keeps_all_points_for_identity_homography():
    # Regression check: a homography with a constant-sign bottom row
    # (e.g. identity) must not filter out any legitimate point.
    H = np.eye(3)
    points = [(1.0, 2.0), (-500.0, 300.0), (0.0, 0.0)]
    assert project_points(H, points) == points
