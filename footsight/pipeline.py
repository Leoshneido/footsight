import argparse

from footsight import pitch_calibration, player_detection, projection, render


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
    ground_points = [projection.bbox_to_ground_point(box) for box in boxes]
    pitch_points = projection.project_points(homography, ground_points)
    pitch_points = projection.filter_to_pitch(pitch_points)

    render.render_pitch(pitch_points, output_path)


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
