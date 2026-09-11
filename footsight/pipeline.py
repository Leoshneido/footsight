import argparse

from footsight import ball_detection, pitch_calibration, player_detection, projection, render, team_classification


def run(
    input_path: str,
    output_path: str,
    detection_model,
    weights_kp: str,
    weights_line: str,
) -> None:
    homography = pitch_calibration.compute_homography(input_path, weights_kp, weights_line)
    if homography is None:
        raise RuntimeError(f"Could not calibrate pitch from {input_path}")

    boxes = player_detection.detect_players(input_path, detection_model)
    labels = team_classification.classify_players(input_path, boxes)

    ground_points = [projection.bbox_to_ground_point(box) for box in boxes]
    projected_indexed = projection.project_points_indexed(homography, ground_points)

    player_positions = []
    for index, point in projected_indexed:
        if not projection.filter_to_pitch([point]):
            continue
        label = labels[index]
        category = "referee" if label == team_classification.OFFICIALS_LABEL else label
        player_positions.append((point, category))

    ball_position = None
    ball_box = ball_detection.find_ball(input_path, boxes)
    if ball_box is not None:
        ball_center = projection.bbox_to_center_point(ball_box)
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
    args = parser.parse_args()

    detection_model = player_detection.load_model()
    run(args.input_path, args.output_path, detection_model, args.weights_kp, args.weights_line)


if __name__ == "__main__":
    main()
