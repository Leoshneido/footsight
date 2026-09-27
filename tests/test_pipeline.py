from unittest.mock import patch

import numpy as np
import pytest

from footsight import pipeline


@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.ball_detection.find_ball")
@patch("footsight.pipeline.team_classification.classify_players")
@patch("footsight.pipeline.player_detection.detect_players")
@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_run_wires_stages_together(
    mock_compute_homography, mock_detect_players, mock_classify_players, mock_detect_ball, mock_render_pitch
):
    mock_compute_homography.return_value = np.eye(3)
    mock_detect_players.return_value = [((0.0, 0.0, 10.0, 20.0), "player")]
    mock_classify_players.return_value = ["team_a"]
    mock_detect_ball.return_value = None

    pipeline.run(
        "still.jpg", "mockup.png",
        detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
    )

    mock_compute_homography.assert_called_once_with("still.jpg", "kp.pt", "lines.pt")
    mock_detect_players.assert_called_once_with("still.jpg", "fake-model")
    mock_classify_players.assert_called_once_with("still.jpg", [(0.0, 0.0, 10.0, 20.0)])
    mock_render_pitch.assert_called_once()
    rendered_points, output_path = mock_render_pitch.call_args[0]
    assert output_path == "mockup.png"
    assert rendered_points == [((5.0, 20.0), "team_a")]
    assert mock_render_pitch.call_args[1] == {"ball_position": None}


@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.ball_detection.find_ball")
@patch("footsight.pipeline.team_classification.classify_players")
@patch("footsight.pipeline.player_detection.detect_players")
@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_run_renders_goalkeepers_and_referees_by_their_detected_role(
    mock_compute_homography, mock_detect_players, mock_classify_players, mock_detect_ball, mock_render_pitch
):
    """Only the boxes the detector called "player" go to the team
    classifier; goalkeepers and officials already know what they are."""
    mock_compute_homography.return_value = np.eye(3)
    # Ground points must land inside the pitch (|x| <= 52.5, |y| <= 34) or
    # the pitch filter drops them before rendering.
    mock_detect_players.return_value = [
        ((0.0, 0.0, 10.0, 20.0), "player"),
        ((30.0, 0.0, 40.0, 20.0), "goalkeeper"),
        ((-50.0, 0.0, -40.0, 20.0), "referee"),
        ((-20.0, 0.0, -10.0, 20.0), "player"),
    ]
    mock_classify_players.return_value = ["team_a", "team_b"]
    mock_detect_ball.return_value = None

    pipeline.run(
        "still.jpg", "mockup.png",
        detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
    )

    mock_classify_players.assert_called_once_with(
        "still.jpg", [(0.0, 0.0, 10.0, 20.0), (-20.0, 0.0, -10.0, 20.0)]
    )
    rendered_points, _ = mock_render_pitch.call_args[0]
    assert rendered_points == [
        ((5.0, 20.0), "team_a"),
        ((35.0, 20.0), "goalkeeper"),
        ((-45.0, 20.0), "referee"),
        ((-15.0, 20.0), "team_b"),
    ]


@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.ball_detection.find_ball")
@patch("footsight.pipeline.team_classification.classify_players")
@patch("footsight.pipeline.player_detection.detect_players")
@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_run_includes_ball_position_when_ball_detected(
    mock_compute_homography, mock_detect_players, mock_classify_players, mock_detect_ball, mock_render_pitch
):
    mock_compute_homography.return_value = np.eye(3)
    mock_detect_players.return_value = []
    mock_classify_players.return_value = []
    mock_detect_ball.return_value = (10.0, 10.0, 20.0, 20.0)  # center = (15.0, 15.0)

    pipeline.run(
        "still.jpg", "mockup.png",
        detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
    )

    mock_render_pitch.assert_called_once_with([], "mockup.png", ball_position=(15.0, 15.0))


@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.ball_detection.find_ball")
@patch("footsight.pipeline.team_classification.classify_players")
@patch("footsight.pipeline.player_detection.detect_players")
@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_run_uses_a_hand_placed_ball_pixel_instead_of_detecting_one(
    mock_compute_homography, mock_detect_players, mock_classify_players, mock_detect_ball, mock_render_pitch
):
    """When the ball is placed by hand -- detection missed it, or picked a
    spare ball by the touchline -- that point wins and detection is skipped
    entirely."""
    mock_compute_homography.return_value = np.eye(3)
    mock_detect_players.return_value = []
    mock_classify_players.return_value = []

    pipeline.run(
        "still.jpg", "mockup.png",
        detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
        ball_pixel=(25.0, 30.0),
    )

    mock_detect_ball.assert_not_called()
    mock_render_pitch.assert_called_once_with([], "mockup.png", ball_position=(25.0, 30.0))


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
@patch("footsight.pipeline.team_classification.classify_players")
@patch("footsight.pipeline.player_detection.detect_players")
@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_run_with_no_players_renders_empty_pitch(
    mock_compute_homography, mock_detect_players, mock_classify_players, mock_detect_ball, mock_render_pitch
):
    mock_compute_homography.return_value = np.eye(3)
    mock_detect_players.return_value = []
    mock_classify_players.return_value = []
    mock_detect_ball.return_value = None

    pipeline.run(
        "still.jpg", "mockup.png",
        detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
    )

    mock_render_pitch.assert_called_once_with([], "mockup.png", ball_position=None)


@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.ball_detection.find_ball")
@patch("footsight.pipeline.team_classification.classify_players")
@patch("footsight.pipeline.player_detection.detect_players")
@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_run_renders_a_trimmed_odd_jersey_as_a_referee(
    mock_compute_homography, mock_detect_players, mock_classify_players, mock_detect_ball, mock_render_pitch
):
    """The classifier trims a jersey hue far from both teams -- typically a
    referee the detector handed over as a player -- and that has to reach
    the mockup as a referee, not as a category render has no color for."""
    mock_compute_homography.return_value = np.eye(3)
    mock_detect_players.return_value = [
        ((0.0, 0.0, 10.0, 20.0), "player"),
        ((30.0, 0.0, 40.0, 20.0), "player"),
    ]
    mock_classify_players.return_value = ["team_a", "officials"]
    mock_detect_ball.return_value = None

    pipeline.run(
        "still.jpg", "mockup.png",
        detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
    )

    rendered_points, _ = mock_render_pitch.call_args[0]
    assert rendered_points == [((5.0, 20.0), "team_a"), ((35.0, 20.0), "referee")]


@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.ball_detection.find_ball")
@patch("footsight.pipeline.team_classification.classify_players")
@patch("footsight.pipeline.player_detection.detect_players")
@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_run_drops_detections_the_user_removed_before_classifying(
    mock_compute_homography, mock_detect_players, mock_classify_players, mock_detect_ball, mock_render_pitch
):
    """A false detection the user removed by hand (the Paramount+ watermark,
    which scores 0.65-0.73 -- as high as a real, partly hidden player) must
    not reach the jersey clustering, where it could skew the team split, nor
    the mockup."""
    mock_compute_homography.return_value = np.eye(3)
    real = ((0.0, 0.0, 10.0, 20.0), "player")
    watermark = ((30.0, 0.0, 40.0, 20.0), "player")
    mock_detect_players.return_value = [real, watermark]
    mock_classify_players.return_value = ["team_a"]
    mock_detect_ball.return_value = None

    reviewed = []

    def review(image_path, detections):
        reviewed.append((image_path, detections))
        return [real]

    pipeline.run(
        "still.jpg", "mockup.png",
        detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
        review=review,
    )

    assert reviewed == [("still.jpg", [real, watermark])]
    mock_classify_players.assert_called_once_with("still.jpg", [(0.0, 0.0, 10.0, 20.0)])
    rendered_points, _ = mock_render_pitch.call_args[0]
    assert rendered_points == [((5.0, 20.0), "team_a")]


@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.ball_detection.find_ball")
@patch("footsight.pipeline.team_classification.classify_players")
@patch("footsight.pipeline.player_detection.detect_players")
@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_run_leaves_off_pitch_detections_out_of_the_team_split(
    mock_compute_homography, mock_detect_players, mock_classify_players, mock_detect_ball, mock_render_pitch
):
    """A ball boy behind the touchline gets detected as a "player" (seen on
    the 11.58.59 and 12.00.14 stills). He was already dropped from the
    mockup, but only after classification -- so his kit still took part in
    the jersey clustering and could skew the team split."""
    mock_compute_homography.return_value = np.eye(3)
    on_pitch = ((0.0, 0.0, 10.0, 20.0), "player")
    ball_boy = ((100.0, 0.0, 110.0, 20.0), "player")  # projects to x=105, past the 57.5 m limit
    mock_detect_players.return_value = [on_pitch, ball_boy]
    mock_classify_players.return_value = ["team_a"]
    mock_detect_ball.return_value = None

    pipeline.run(
        "still.jpg", "mockup.png",
        detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
    )

    mock_classify_players.assert_called_once_with("still.jpg", [(0.0, 0.0, 10.0, 20.0)])
    rendered_points, _ = mock_render_pitch.call_args[0]
    assert rendered_points == [((5.0, 20.0), "team_a")]
