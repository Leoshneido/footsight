import pytest

from footsight.ball_picker import display_scale, to_source_pixel


def test_display_scale_shrinks_a_broadcast_still_to_fit_the_screen():
    assert display_scale((3020, 1700), max_width_px=1400) == pytest.approx(1400 / 3020)


def test_display_scale_never_enlarges_a_small_still():
    assert display_scale((800, 450), max_width_px=1400) == 1.0


def test_to_source_pixel_maps_a_click_back_to_full_resolution():
    """The picker shows a shrunken still, so a click lands in display
    coordinates -- the ball's real pixel is what the homography needs."""
    assert to_source_pixel((350.0, 200.0), scale=0.5) == (700.0, 400.0)


def test_to_source_pixel_is_identity_at_full_size():
    assert to_source_pixel((120.0, 80.0), scale=1.0) == (120.0, 80.0)
