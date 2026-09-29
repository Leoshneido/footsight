from unittest.mock import patch

import numpy as np
import pytest

from footsight import pipeline


@pytest.fixture(autouse=True)
def no_camera_view():
    """The camera view renders from the real still; these tests use fake
    paths, so it is stubbed unless a test inspects it."""
    with patch("footsight.pipeline.camera_view.render_camera_view") as mock:
        yield mock


@pytest.fixture(autouse=True)
def officials_in_black():
    """official_kits reads the still from disk; these tests use fake paths,
    so every official "wears" black unless a test says otherwise."""
    with patch("footsight.pipeline.team_classification.official_kits",
               side_effect=lambda path, boxes: [((30, 40, 30), (20, 20, 20))] * len(boxes)) as mock:
        yield mock


@pytest.fixture(autouse=True)
def no_kit_colors():
    """team_kit_colors reads the still from disk; these tests use fake
    paths, so it is stubbed to "no detected kits" unless a test says
    otherwise."""
    with patch("footsight.pipeline.team_classification.team_kit_colors", return_value={}) as mock:
        yield mock


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
    assert mock_render_pitch.call_args[1] == {"ball_position": None, "kit_colors": {}, "player_kits": [None]}


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
    # The ball touches the grass at the bottom of its box, not its center --
    # projecting the center put it at a player's shins instead of their feet.
    mock_detect_ball.return_value = (10.0, 10.0, 20.0, 20.0)  # bottom-center = (15.0, 20.0)

    pipeline.run(
        "still.jpg", "mockup.png",
        detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
    )

    mock_render_pitch.assert_called_once_with([], "mockup.png", ball_position=(15.0, 20.0), kit_colors={}, player_kits=[])


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
    mock_render_pitch.assert_called_once_with([], "mockup.png", ball_position=(25.0, 30.0), kit_colors={}, player_kits=[])


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

    mock_render_pitch.assert_called_once_with([], "mockup.png", ball_position=None, kit_colors={}, player_kits=[])


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



@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.ball_detection.find_ball")
@patch("footsight.pipeline.team_classification.classify_players")
@patch("footsight.pipeline.player_detection.detect_players")
@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_run_draws_teams_in_the_kit_colors_read_off_the_still(
    mock_compute_homography, mock_detect_players, mock_classify_players, mock_detect_ball, mock_render_pitch,
    no_kit_colors,
):
    mock_compute_homography.return_value = np.eye(3)
    mock_detect_players.return_value = [
        ((0.0, 0.0, 10.0, 20.0), "player"),
        ((30.0, 0.0, 40.0, 20.0), "player"),
        ((100.0, 0.0, 110.0, 20.0), "player"),  # off the pitch
    ]
    mock_classify_players.return_value = ["team_a", "team_b"]
    mock_detect_ball.return_value = None
    kits = {"team_a": ((245, 130, 30), (240, 240, 240)), "team_b": ((30, 30, 35), (30, 30, 35))}
    no_kit_colors.return_value = kits

    pipeline.run(
        "still.jpg", "mockup.png",
        detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
    )

    no_kit_colors.assert_called_once_with(
        "still.jpg", [(0.0, 0.0, 10.0, 20.0), (30.0, 0.0, 40.0, 20.0)], ["team_a", "team_b"]
    )
    assert mock_render_pitch.call_args[1]["kit_colors"] == kits


@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.ball_detection.find_ball")
@patch("footsight.pipeline.team_classification.classify_players")
@patch("footsight.pipeline.player_detection.detect_players")
@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_run_lets_the_user_remove_a_wrongly_detected_ball(
    mock_compute_homography, mock_detect_players, mock_classify_players, mock_detect_ball, mock_render_pitch
):
    mock_compute_homography.return_value = np.eye(3)
    mock_detect_players.return_value = []
    mock_classify_players.return_value = []
    mock_detect_ball.return_value = (10.0, 10.0, 20.0, 20.0)  # bottom-center = (15.0, 20.0)
    reviewed = []

    def ball_review(image_path, detected):
        reviewed.append((image_path, detected))
        return None

    pipeline.run(
        "still.jpg", "mockup.png",
        detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
        ball_review=ball_review,
    )

    assert reviewed == [("still.jpg", (15.0, 20.0))]
    assert mock_render_pitch.call_args[1]["ball_position"] is None


@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.ball_detection.find_ball")
@patch("footsight.pipeline.team_classification.classify_players")
@patch("footsight.pipeline.player_detection.detect_players")
@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_run_lets_the_user_add_a_ball_detection_missed(
    mock_compute_homography, mock_detect_players, mock_classify_players, mock_detect_ball, mock_render_pitch
):
    mock_compute_homography.return_value = np.eye(3)
    mock_detect_players.return_value = []
    mock_classify_players.return_value = []
    mock_detect_ball.return_value = None
    reviewed = []

    def ball_review(image_path, detected):
        reviewed.append(detected)
        return (25.0, 30.0)

    pipeline.run(
        "still.jpg", "mockup.png",
        detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
        ball_review=ball_review,
    )

    assert reviewed == [None]
    assert mock_render_pitch.call_args[1]["ball_position"] == (25.0, 30.0)



def test_camera_output_path_sits_next_to_the_top_down_mockup():
    assert pipeline.camera_output_path("out/mockup.png") == "out/mockup_camera.png"
    assert pipeline.camera_output_path("mockup.png") == "mockup_camera.png"


@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.ball_detection.find_ball")
@patch("footsight.pipeline.team_classification.classify_players")
@patch("footsight.pipeline.player_detection.detect_players")
@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_run_also_writes_the_camera_view_with_the_same_people(
    mock_compute_homography, mock_detect_players, mock_classify_players, mock_detect_ball, mock_render_pitch,
    no_camera_view, no_kit_colors,
):
    """One run, two images: the camera view gets the same on-pitch people,
    categories and kit colors as the top-down mockup, and the same ball."""
    mock_compute_homography.return_value = np.eye(3)
    mock_detect_players.return_value = [
        ((0.0, 0.0, 10.0, 20.0), "player"),
        ((30.0, 0.0, 40.0, 20.0), "goalkeeper"),
        ((100.0, 0.0, 110.0, 20.0), "player"),  # off the pitch
    ]
    mock_classify_players.return_value = ["team_b"]
    mock_detect_ball.return_value = (10.0, 10.0, 20.0, 20.0)
    no_kit_colors.return_value = {"team_b": ((30, 30, 35), (200, 20, 40))}

    pipeline.run(
        "still.jpg", "out/mockup.png",
        detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
    )

    no_camera_view.assert_called_once()
    image_path, homography, people, ball_pixel, output_path = no_camera_view.call_args[0]
    assert image_path == "still.jpg" and output_path == "out/mockup_camera.png"
    assert [(box, category) for box, category, _, _ in people] == [
        ((0.0, 0.0, 10.0, 20.0), "team_b"),
        ((30.0, 0.0, 40.0, 20.0), "goalkeeper"),
    ]
    from footsight.render import category_kit
    assert people[0][2] == category_kit("team_b", no_kit_colors.return_value)
    assert [pose for *_, pose in people] == [None, None]  # no pose model given
    assert ball_pixel == (15.0, 20.0)


@patch("footsight.pipeline.pose.estimate_poses")
@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.ball_detection.find_ball")
@patch("footsight.pipeline.team_classification.classify_players")
@patch("footsight.pipeline.player_detection.detect_players")
@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_run_poses_only_the_people_on_the_pitch(
    mock_compute_homography, mock_detect_players, mock_classify_players, mock_detect_ball, mock_render_pitch,
    mock_estimate_poses, no_camera_view,
):
    mock_compute_homography.return_value = np.eye(3)
    mock_detect_players.return_value = [
        ((0.0, 0.0, 10.0, 20.0), "player"),
        ((100.0, 0.0, 110.0, 20.0), "player"),  # off the pitch
    ]
    mock_classify_players.return_value = ["team_a"]
    mock_detect_ball.return_value = None
    pose = np.zeros((17, 3))
    mock_estimate_poses.return_value = [pose]

    pipeline.run(
        "still.jpg", "mockup.png",
        detection_model="fake-model", weights_kp="kp.pt", weights_line="lines.pt",
        pose_model="fake-pose-model",
    )

    mock_estimate_poses.assert_called_once_with("still.jpg", [(0.0, 0.0, 10.0, 20.0)], "fake-pose-model")
    people = no_camera_view.call_args[0][2]
    # the pose goes through the JSON analysis cache, so it arrives as an equal copy
    assert np.array_equal(people[0][3], pose)


# ---- analyze / render split (capture studio) ----

def _analysis(detections, ball_pixel=None, poses=None):
    return {
        "version": 1,
        "homography": np.eye(3).tolist(),
        "detections": [{"id": i, "box": list(box), "role": role} for i, (box, role) in enumerate(detections)],
        "poses": poses or {},
        "ball_pixel": ball_pixel,
    }


@patch("footsight.pipeline.pose.estimate_poses")
@patch("footsight.pipeline.ball_detection.find_ball")
@patch("footsight.pipeline.player_detection.detect_players")
@patch("footsight.pipeline.pitch_calibration.compute_homography")
def test_analyze_gives_each_detection_a_stable_id_and_can_be_saved(
    mock_compute_homography, mock_detect_players, mock_detect_ball, mock_estimate_poses
):
    """The slow step's results are cached as JSON, and every detection keeps
    the number the detector gave it, so fixes and drawings can refer to it."""
    import json

    mock_compute_homography.return_value = np.eye(3)
    mock_detect_players.return_value = [
        ((0.0, 0.0, 10.0, 20.0), "player"),
        ((100.0, 0.0, 110.0, 20.0), "player"),  # off the pitch: not posed
        ((30.0, 0.0, 40.0, 20.0), "goalkeeper"),
    ]
    mock_detect_ball.return_value = (10.0, 10.0, 20.0, 20.0)
    mock_estimate_poses.return_value = [np.zeros((17, 3)), None]

    analysis = pipeline.analyze("still.jpg", "det", "kp.pt", "lines.pt", pose_model="pose")

    saved = json.loads(json.dumps(analysis))
    assert [(d["id"], d["role"]) for d in saved["detections"]] == [(0, "player"), (1, "player"), (2, "goalkeeper")]
    assert set(saved["poses"]) == {"0", "2"}
    assert saved["poses"]["2"] is None
    assert saved["ball_pixel"] == [15.0, 20.0]
    mock_estimate_poses.assert_called_once_with("still.jpg", [(0.0, 0.0, 10.0, 20.0), (30.0, 0.0, 40.0, 20.0)], "pose")


@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.team_classification.classify_players")
def test_render_leaves_out_players_the_user_removed(mock_classify_players, mock_render_pitch, no_camera_view):
    mock_classify_players.return_value = ["team_a"]
    analysis = _analysis([((0.0, 0.0, 10.0, 20.0), "player"), ((30.0, 0.0, 40.0, 20.0), "player")])

    pipeline.render_still("still.jpg", analysis, "mockup.png", {"removed": [1]})

    mock_classify_players.assert_called_once_with("still.jpg", [(0.0, 0.0, 10.0, 20.0)])
    assert no_camera_view.call_args.kwargs["player_ids"] == [0]


@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.team_classification.classify_players")
def test_render_adds_a_player_the_detector_missed(mock_classify_players, mock_render_pitch, no_camera_view):
    """A player hidden behind another (the Bayern still): the user clicks
    their feet and picks a side. They're drawn in that kit, stand 1.8 m tall
    on the player-height scale, and stay out of the team split."""
    mock_classify_players.return_value = ["team_a", "team_a"]
    # box heights: 20 px with feet on row 20, 30 px on row 30 -> 10 px on row 10
    analysis = _analysis([((0.0, 0.0, 10.0, 20.0), "player"), ((30.0, 0.0, 40.0, 30.0), "player")])
    corrections = {"added": [{"id": 1000, "feet": [20.0, 10.0], "category": "team_b"}]}

    pipeline.render_still("still.jpg", analysis, "mockup.png", corrections)

    mock_classify_players.assert_called_once_with("still.jpg", [(0.0, 0.0, 10.0, 20.0), (30.0, 0.0, 40.0, 30.0)])
    positions, _ = mock_render_pitch.call_args[0]
    assert ((20.0, 10.0), "team_b") in positions
    people = no_camera_view.call_args[0][2]
    box, category, _, pose = people[-1]
    assert category == "team_b" and pose is None
    assert box == pytest.approx((18.0, 0.0, 22.0, 10.0))
    assert no_camera_view.call_args.kwargs["player_ids"] == [0, 1, 1000]


@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.team_classification.classify_players")
def test_render_places_or_removes_the_ball_as_corrected(mock_classify_players, mock_render_pitch, no_camera_view):
    mock_classify_players.return_value = []
    analysis = _analysis([], ball_pixel=[15.0, 20.0])

    pipeline.render_still("still.jpg", analysis, "mockup.png")
    assert mock_render_pitch.call_args.kwargs["ball_position"] == (15.0, 20.0)

    pipeline.render_still("still.jpg", analysis, "mockup.png", {"ball": {"mode": "set", "pixel": [5.0, 6.0]}})
    assert mock_render_pitch.call_args.kwargs["ball_position"] == (5.0, 6.0)
    assert no_camera_view.call_args[0][3] == (5.0, 6.0)

    pipeline.render_still("still.jpg", analysis, "mockup.png", {"ball": {"mode": "none"}})
    assert mock_render_pitch.call_args.kwargs["ball_position"] is None
    assert no_camera_view.call_args[0][3] is None


@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.team_classification.classify_players")
def test_render_gives_the_editor_every_kit_for_the_add_player_chooser(mock_classify_players, mock_render_pitch, no_camera_view, no_kit_colors):
    from footsight.render import category_kit

    mock_classify_players.return_value = []
    no_kit_colors.return_value = {"team_a": ((245, 130, 30), (240, 240, 240))}

    pipeline.render_still("still.jpg", _analysis([]), "mockup.png")

    kits = no_camera_view.call_args.kwargs["kits"]
    assert set(kits) >= {"team_a", "team_b", "goalkeeper", "referee"}
    assert kits["team_a"] == category_kit("team_a", no_kit_colors.return_value)


@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.team_classification.classify_players")
def test_render_puts_a_player_on_the_side_the_user_chose(mock_classify_players, mock_render_pitch, no_camera_view):
    """Color can't fix everything: the detector sometimes calls a player a
    referee (player 20 on the Colombia v Portugal capture). The user's side
    wins, for detected and added players alike."""
    mock_classify_players.return_value = ["team_a"]
    analysis = _analysis([((0.0, 0.0, 10.0, 20.0), "player"), ((30.0, 0.0, 40.0, 20.0), "referee")])
    corrections = {"sides": {"1": "team_b", "0": "goalkeeper"}}

    pipeline.render_still("still.jpg", analysis, "mockup.png", corrections)

    positions, _ = mock_render_pitch.call_args[0]
    assert [category for _, category in positions] == ["goalkeeper", "team_b"]
    assert [category for _, category, _, _ in no_camera_view.call_args[0][2]] == ["goalkeeper", "team_b"]


@patch("footsight.pipeline.team_classification.official_kits")
@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.team_classification.classify_players")
def test_officials_are_drawn_in_the_kit_they_actually_wear(mock_classify_players, mock_render_pitch, mock_official_kits, no_camera_view):
    """Officials were all drawn in one fixed yellow -- almost Colombia's
    drawn yellow, so the (black-kitted) referee looked like a Colombian."""
    mock_classify_players.return_value = ["team_a"]
    mock_official_kits.return_value = [((30, 40, 30), (20, 20, 20))]
    analysis = _analysis([((0.0, 0.0, 10.0, 20.0), "player"), ((30.0, 0.0, 40.0, 20.0), "referee")])

    pipeline.render_still("still.jpg", analysis, "mockup.png")

    mock_official_kits.assert_called_once_with("still.jpg", [(30.0, 0.0, 40.0, 20.0)])
    people = no_camera_view.call_args[0][2]
    assert people[1][1] == "referee" and people[1][2] == ((30, 40, 30), (20, 20, 20))
    assert mock_render_pitch.call_args.kwargs["player_kits"] == [None, ((30, 40, 30), (20, 20, 20))]


@patch("footsight.pipeline.team_classification.official_kits")
@patch("footsight.pipeline.render.render_pitch")
@patch("footsight.pipeline.team_classification.classify_players")
def test_an_official_dressed_like_a_team_is_drawn_neutral(mock_classify_players, mock_render_pitch, mock_official_kits, no_camera_view, no_kit_colors):
    mock_classify_players.return_value = ["team_a"]
    no_kit_colors.return_value = {"team_a": ((202, 179, 0), (240, 240, 240)), "team_b": ((108, 40, 20), (60, 20, 20))}
    mock_official_kits.return_value = [((210, 185, 10), (20, 20, 20))]  # a yellow referee next to a yellow team
    analysis = _analysis([((0.0, 0.0, 10.0, 20.0), "player"), ((30.0, 0.0, 40.0, 20.0), "referee")])

    pipeline.render_still("still.jpg", analysis, "mockup.png")

    assert no_camera_view.call_args[0][2][1][2] == pipeline.NEUTRAL_OFFICIAL_KIT
