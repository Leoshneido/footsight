# tests/test_pitch_calibration.py
from pathlib import Path

import numpy as np

from footsight.pitch_calibration import compute_homography

FIXTURE = Path(__file__).parent / "fixtures" / "sample_still.png"


def test_compute_homography_returns_invertible_3x3_or_none():
    # Integration test: runs the real PnLCalib subprocess against a real still.
    # Requires Task 5's setup steps (submodule + weights) to have been run.
    # Can take up to ~60s on CPU.
    homography = compute_homography(str(FIXTURE))

    if homography is None:
        return  # acceptable per the spec's error-handling section

    assert homography.shape == (3, 3)
    assert np.all(np.isfinite(homography))
    np.linalg.inv(homography)  # raises if degenerate/non-invertible
