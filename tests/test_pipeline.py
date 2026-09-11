from unittest.mock import patch

import numpy as np
import pytest

from footsight import pipeline


@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.ball_detection.find_ball")
@patch("footsight.pipeline.player_detection.detect_players")
@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_run_wires_stages_together(
    mock_compute_homography, mock_detect_players, mock_detect_ball, mock_render_pitch
):
    mock_compute_homography.return_value = np.eye(3)
    mock_detect_players.return_value = [(0.0, 0.0, 10.0, 20.0)]
    mock_detect_ball.return_value = None

    pipeline.run(
        "still.jpg", "mockup.png",
        detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
    )

    mock_compute_homography.assert_called_once_with("still.jpg", "kp.pt", "lines.pt")
    mock_detect_players.assert_called_once_with("still.jpg", "fake-model")
    mock_detect_ball.assert_called_once_with("still.jpg", [(0.0, 0.0, 10.0, 20.0)])
    mock_render_pitch.assert_called_once()
    rendered_points, output_path = mock_render_pitch.call_args[0]
    assert output_path == "mockup.png"
    assert rendered_points == [(5.0, 20.0)]
    assert mock_render_pitch.call_args[1] == {"ball_position": None}


@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.ball_detection.find_ball")
@patch("footsight.pipeline.player_detection.detect_players")
@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_run_includes_ball_position_when_ball_detected(
    mock_compute_homography, mock_detect_players, mock_detect_ball, mock_render_pitch
):
    mock_compute_homography.return_value = np.eye(3)
    mock_detect_players.return_value = []
    mock_detect_ball.return_value = (10.0, 10.0, 20.0, 20.0)  # center = (15.0, 15.0)

    pipeline.run(
        "still.jpg", "mockup.png",
        detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
    )

    mock_render_pitch.assert_called_once_with([], "mockup.png", ball_position=(15.0, 15.0))


@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_run_raises_when_calibration_fails(mock_compute_homography):
    mock_compute_homography.return_value = None

    with pytest.raises(RuntimeError, match="Could not calibrate"):
        pipeline.run(
            "still.jpg", "mockup.png",
            detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
        )


@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.ball_detection.find_ball")
@patch("footsight.pipeline.player_detection.detect_players")
@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_run_with_no_players_renders_empty_pitch(
    mock_compute_homography, mock_detect_players, mock_detect_ball, mock_render_pitch
):
    mock_compute_homography.return_value = np.eye(3)
    mock_detect_players.return_value = []
    mock_detect_ball.return_value = None

    pipeline.run(
        "still.jpg", "mockup.png",
        detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
    )

    mock_render_pitch.assert_called_once_with([], "mockup.png", ball_position=None)
