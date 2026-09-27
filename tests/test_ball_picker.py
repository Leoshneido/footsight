import pytest

from footsight.ball_picker import display_scale, handle_key, to_source_pixel


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


ENTER, ESC, DELETE, BACKSPACE = 13, 27, 127, 8


def test_handle_key_enter_accepts_the_current_ball():
    assert handle_key(ENTER, current=(700.0, 400.0), detected=(10.0, 10.0)) == (True, (700.0, 400.0))


def test_handle_key_enter_accepts_no_ball_after_it_was_removed():
    assert handle_key(ENTER, current=None, detected=(10.0, 10.0)) == (True, None)


def test_handle_key_escape_keeps_whatever_detection_found():
    assert handle_key(ESC, current=(700.0, 400.0), detected=(10.0, 10.0)) == (True, (10.0, 10.0))
    assert handle_key(ESC, current=(700.0, 400.0), detected=None) == (True, None)


def test_handle_key_delete_or_backspace_removes_a_wrong_ball():
    """A spare ball by the touchline, or a sock, detected as the ball -- the
    user takes it off the mockup entirely."""
    assert handle_key(DELETE, current=(10.0, 10.0), detected=(10.0, 10.0)) == (False, None)
    assert handle_key(BACKSPACE, current=(10.0, 10.0), detected=(10.0, 10.0)) == (False, None)


def test_handle_key_ignores_other_keys():
    assert handle_key(ord("a"), current=(10.0, 10.0), detected=None) == (False, (10.0, 10.0))
