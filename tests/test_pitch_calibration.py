# tests/test_pitch_calibration.py
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from footsight.pitch_calibration import compute_homography

FIXTURE = Path(__file__).parent / "fixtures" / "sample_still.png"


@patch("footsight.pitch_calibration.subprocess.run")
def test_compute_homography_raises_runtime_error_with_stderr_on_subprocess_failure(mock_run):
    mock_run.return_value = MagicMock(
        returncode=1, stdout="", stderr="Traceback: something broke in PnLCalib"
    )
    with pytest.raises(RuntimeError, match="something broke in PnLCalib"):
        compute_homography("fake.png")


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
