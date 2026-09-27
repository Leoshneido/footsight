import argparse

from footsight import (
    ball_detection,
    ball_picker,
    detection_picker,
    pitch_calibration,
    player_detection,
    projection,
    render,
    team_classification,
)


def run(
    input_path: str,
    output_path: str,
    detection_model,
    weights_kp: str,
    weights_line: str,
    ball_pixel: tuple[float, float] | None = None,
    review=None,
) -> None:
    homography = pitch_calibration.compute_homography(input_path, weights_kp, weights_line)
    if homography is None:
        raise RuntimeError(f"Could not calibrate pitch from {input_path}")

    detections = player_detection.detect_players(input_path, detection_model)
    # Hand-removed false detections have to go before clustering: a
    # watermark counted as a player can skew the team split.
    if review is not None:
        detections = review(input_path, detections)
    boxes = [box for box, _ in detections]
    player_boxes = [box for box, role in detections if role == player_detection.PLAYER_ROLE]
    team_labels = iter(team_classification.classify_players(input_path, player_boxes))

    # Goalkeepers and officials are rendered by the role the detector gave
    # them; only outfield players need a team worked out from jersey color.
    # A player the classifier trimmed as an odd jersey color is a referee
    # the detector mislabeled, so it renders as one.
    categories = []
    for _, role in detections:
        if role != player_detection.PLAYER_ROLE:
            categories.append(role)
            continue
        label = next(team_labels)
        categories.append("referee" if label == team_classification.OFFICIALS_LABEL else label)

    ground_points = [projection.bbox_to_ground_point(box) for box in boxes]
    projected_indexed = projection.project_points_indexed(homography, ground_points)

    player_positions = []
    for index, point in projected_indexed:
        if not projection.filter_to_pitch([point]):
            continue
        player_positions.append((point, categories[index]))

    ball_position = None
    # A hand-placed ball wins outright: it is only ever supplied because
    # detection got it wrong.
    ball_box = None if ball_pixel is not None else ball_detection.find_ball(input_path, boxes)
    if ball_pixel is not None:
        ball_center = ball_pixel
    elif ball_box is not None:
        ball_center = projection.bbox_to_center_point(ball_box)
    else:
        ball_center = None

    if ball_center is not None:
        projected_ball = projection.filter_to_pitch(projection.project_points(homography, [ball_center]))
        if projected_ball:
            ball_position = projected_ball[0]

    render.render_pitch(player_positions, output_path, ball_position=ball_position)


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn a broadcast still into a 2D pitch mockup.")
    parser.add_argument("input_path", help="Path to the input still image")
    parser.add_argument("output_path", help="Path to save the rendered mockup PNG")
    parser.add_argument("--weights-kp", default=str(pitch_calibration.DEFAULT_WEIGHTS_KP))
    parser.add_argument("--weights-line", default=str(pitch_calibration.DEFAULT_WEIGHTS_LINE))
    parser.add_argument("--player-weights", default=str(player_detection.DEFAULT_WEIGHTS))
    parser.add_argument(
        "--pick-ball",
        action="store_true",
        help="Click the ball on the still yourself instead of detecting it",
    )
    parser.add_argument(
        "--remove-detections",
        action="store_true",
        help="Review the detections on the still and click false ones to remove them",
    )
    args = parser.parse_args()

    ball_pixel = ball_picker.pick_ball_pixel(args.input_path) if args.pick_ball else None

    detection_model = player_detection.load_model(args.player_weights)
    run(
        args.input_path,
        args.output_path,
        detection_model,
        args.weights_kp,
        args.weights_line,
        ball_pixel=ball_pixel,
        review=detection_picker.review_detections if args.remove_detections else None,
    )


if __name__ == "__main__":
    main()
