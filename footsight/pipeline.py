import argparse
from pathlib import Path

import numpy as np

from footsight import (
    ball_detection,
    ball_picker,
    camera_view,
    detection_picker,
    pitch_calibration,
    player_detection,
    pose,
    projection,
    render,
    team_classification,
)


def camera_output_path(output_path: str) -> str:
    """Where the camera-angle view goes: next to the top-down mockup, with
    "_camera" before the extension."""
    path = Path(output_path)
    return str(path.with_name(f"{path.stem}_camera{path.suffix}"))


ADDED_ID_START = 1000
OFFICIAL_CATEGORIES = ("referee", "assistant_referee")
# Officials are drawn in the kit they actually wear, unless it would look like
# a team's (a yellow referee next to a yellow team): then this neutral one.
NEUTRAL_OFFICIAL_KIT = ((62, 64, 70), (28, 28, 30))
ADDED_BOX_WIDTH = 0.4  # an added player's box: this fraction of its height wide


def analyze(input_path: str, detection_model, weights_kp: str, weights_line: str, pose_model=None, detect_ball: bool = True) -> dict:
    """The slow part of processing a still -- calibration, detection, poses
    and the automatic ball -- as a JSON-ready dict, so fixes can re-render
    from it in seconds. Every detection keeps a stable id (its position in
    the detector's output) that corrections and editor drawings refer to."""
    homography = pitch_calibration.compute_homography(input_path, weights_kp, weights_line)
    if homography is None:
        raise RuntimeError(f"Could not calibrate pitch from {input_path}")

    detections = [
        {"id": index, "box": [float(v) for v in box], "role": role}
        for index, (box, role) in enumerate(player_detection.detect_players(input_path, detection_model))
    ]
    on_pitch = _on_pitch(homography, [(d, _box(d)) for d in detections])

    poses = {}
    if pose_model is not None:
        estimated = pose.estimate_poses(input_path, [box for _, (_, box) in on_pitch], pose_model)
        poses = {str(d["id"]): (None if p is None else np.asarray(p).tolist()) for (_, (d, _)), p in zip(on_pitch, estimated)}

    ball_pixel = None
    if detect_ball:
        ball_box = ball_detection.find_ball(input_path, [_box(d) for d in detections])
        if ball_box is not None:
            # where the ball touches the grass (bottom of its box), like a
            # player's feet -- its center sits a ball-radius above the pitch
            ball_pixel = [float(v) for v in projection.bbox_to_ground_point(ball_box)]

    return {
        "version": 1,
        "homography": np.asarray(homography, dtype=float).tolist(),
        "detections": detections,
        "poses": poses,
        "ball_pixel": ball_pixel,
    }


def _box(detection: dict) -> tuple[float, float, float, float]:
    return tuple(float(v) for v in detection["box"])


def _on_pitch(homography, items):
    """(pitch point, item) for items whose ground point lands on the pitch;
    items are (anything, box) pairs. Off-pitch people (a ball boy behind the
    touchline) never reach the team split, where they could skew it."""
    ground = [projection.bbox_to_ground_point(box) for _, box in items]
    return [
        (point, items[index])
        for index, point in projection.project_points_indexed(homography, ground)
        if projection.filter_to_pitch([point])
    ]


def normalize_corrections(corrections: dict | None) -> dict:
    corrections = corrections or {}
    return {
        "removed": [int(i) for i in corrections.get("removed", [])],
        "added": [
            {"id": int(a["id"]), "feet": [float(v) for v in a["feet"]], "category": a["category"]}
            for a in corrections.get("added", [])
        ],
        "ball": corrections.get("ball") or {"mode": "auto"},
        # the user's side for a detected player, by id (JSON keys are strings)
        "sides": {str(k): v for k, v in (corrections.get("sides") or {}).items()},
    }


def render_still(input_path: str, analysis: dict, output_path: str, corrections: dict | None = None) -> None:
    """The fast part: apply corrections, split teams, and draw the top-down
    mockup (output_path) and the camera view next to it."""
    homography = np.asarray(analysis["homography"], dtype=float)
    fixes = normalize_corrections(corrections)
    removed = set(fixes["removed"])
    kept = [d for d in analysis["detections"] if d["id"] not in removed]
    on_pitch = _on_pitch(homography, [(d, _box(d)) for d in kept])

    player_boxes = [box for _, (d, box) in on_pitch if d["role"] == player_detection.PLAYER_ROLE]
    player_labels = team_classification.classify_players(input_path, player_boxes)
    kit_colors = team_classification.team_kit_colors(input_path, player_boxes, player_labels)
    team_labels = iter(player_labels)

    # Goalkeepers and officials are rendered by the role the detector gave
    # them; only outfield players need a team worked out from jersey color.
    # A player the classifier trimmed as an odd jersey color is a referee
    # the detector mislabeled, so it renders as one.
    player_positions, people, ids = [], [], []
    poses = {str(k): v for k, v in (analysis.get("poses") or {}).items()}
    for point, (d, box) in on_pitch:
        if d["role"] != player_detection.PLAYER_ROLE:
            category = d["role"]
        else:
            label = next(team_labels)
            category = "referee" if label == team_classification.OFFICIALS_LABEL else label
        category = fixes["sides"].get(str(d["id"]), category)  # the user's side wins
        player_pose = poses.get(str(d["id"]))
        player_positions.append((point, category))
        people.append((box, category, render.category_kit(category, kit_colors), None if player_pose is None else np.asarray(player_pose)))
        ids.append(d["id"])

    # Players the detector missed (hidden behind a teammate): placed by hand,
    # in the side the user picked, never part of the team split.
    if fixes["added"]:
        heights = camera_view.vertical_scale([box for _, (_, box) in on_pitch], homography, _image_width(input_path, on_pitch))
        for added in fixes["added"]:
            x, y = added["feet"]
            h = float(camera_view.PLAYER_HEIGHT_M * heights(y))
            w = ADDED_BOX_WIDTH * h
            box = (x - w / 2, y - h, x + w / 2, y)
            for point, _ in _on_pitch(homography, [(added, box)]):
                player_positions.append((point, added["category"]))
                people.append((box, added["category"], render.category_kit(added["category"], kit_colors), None))
                ids.append(added["id"])

    player_kits = _dress_officials(input_path, people, ids, kit_colors)
    people = [(box, category, kit or colors, pose) for (box, category, colors, pose), kit in zip(people, player_kits)]

    ball = fixes["ball"]
    if ball.get("mode") == "none":
        ball_pixel = None
    elif ball.get("mode") == "set":
        ball_pixel = tuple(float(v) for v in ball["pixel"])
    else:
        ball_pixel = None if analysis.get("ball_pixel") is None else tuple(float(v) for v in analysis["ball_pixel"])
    ball_position = None
    if ball_pixel is not None:
        projected = projection.filter_to_pitch(projection.project_points(homography, [ball_pixel]))
        ball_position = projected[0] if projected else None

    render.render_pitch(player_positions, output_path, ball_position=ball_position, kit_colors=kit_colors, player_kits=player_kits)
    camera_view.render_camera_view(
        input_path, homography, people, ball_pixel if ball_position is not None else None, camera_output_path(output_path),
        player_ids=ids, kits={category: render.category_kit(category, kit_colors) for category in render.CATEGORY_COLORS},
    )


def _dress_officials(input_path: str, people, ids, kit_colors) -> list:
    """Per-person (shirt, shorts) for officials, None for everyone else.
    Detected officials get the colors read off the still; hand-added ones
    (no reliable pixels) and any whose kit looks like a team's get the
    neutral kit, so an official never passes for a team player."""
    kits = [None] * len(people)
    officials = [i for i, (_, category, _, _) in enumerate(people) if category in OFFICIAL_CATEGORIES]
    if not officials:
        return kits
    detected = [i for i in officials if ids[i] < ADDED_ID_START]
    worn = dict(zip(detected, team_classification.official_kits(input_path, [people[i][0] for i in detected]))) if detected else {}
    team_shirts = [
        kit_colors[team][0] if team in kit_colors else fixed
        for team, fixed in (("team_a", render.TEAM_A_COLOR), ("team_b", render.TEAM_B_COLOR))
    ]
    for i in officials:
        kit = worn.get(i)
        if kit is None or any(np.linalg.norm(np.subtract(kit[0], shirt)) < team_classification.KIT_MIN_DISTANCE for shirt in team_shirts):
            kit = NEUTRAL_OFFICIAL_KIT
        kits[i] = (tuple(int(c) for c in kit[0]), tuple(int(c) for c in kit[1]))
    return kits


def _image_width(input_path: str, on_pitch) -> float:
    """Only needed when too few players are detected to fit the height scale."""
    if len(on_pitch) >= 2:
        return 0.0
    from PIL import Image

    return float(Image.open(input_path).size[0])


def run(
    input_path: str,
    output_path: str,
    detection_model,
    weights_kp: str,
    weights_line: str,
    ball_pixel: tuple[float, float] | None = None,
    review=None,
    ball_review=None,
    pose_model=None,
) -> None:
    """Write the top-down mockup to output_path and the camera-angle view
    next to it (camera_output_path): analyze, apply the optional hand
    reviews as corrections, render. Without a pose_model every player in
    the camera view is drawn as the standing figure."""
    analysis = analyze(input_path, detection_model, weights_kp, weights_line, pose_model, detect_ball=ball_pixel is None)
    corrections = {"removed": [], "added": [], "ball": {"mode": "auto"}}

    # Hand-removed false detections have to go before clustering: a
    # watermark counted as a player can skew the team split.
    if review is not None:
        detections = [(_box(d), d["role"]) for d in analysis["detections"]]
        kept = review(input_path, detections)
        corrections["removed"] = [d["id"] for d, pair in zip(analysis["detections"], detections) if pair not in kept]

    # A hand-placed ball wins outright; the ball review can move, add or
    # remove (None) what detection found.
    ball = tuple(ball_pixel) if ball_pixel is not None else (tuple(analysis["ball_pixel"]) if analysis["ball_pixel"] else None)
    if ball_pixel is not None:
        corrections["ball"] = {"mode": "set", "pixel": list(ball)}
    if ball_review is not None:
        reviewed = ball_review(input_path, ball)
        corrections["ball"] = {"mode": "none"} if reviewed is None else {"mode": "set", "pixel": list(reviewed)}

    render_still(input_path, analysis, output_path, corrections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn a broadcast still into a 2D pitch mockup.")
    parser.add_argument("input_path", help="Path to the input still image")
    parser.add_argument("output_path", help="Path to save the rendered mockup PNG")
    parser.add_argument("--weights-kp", default=str(pitch_calibration.DEFAULT_WEIGHTS_KP))
    parser.add_argument("--weights-line", default=str(pitch_calibration.DEFAULT_WEIGHTS_LINE))
    parser.add_argument("--player-weights", default=str(player_detection.DEFAULT_WEIGHTS))
    parser.add_argument("--pose-weights", default=str(pose.DEFAULT_POSE_WEIGHTS))
    parser.add_argument(
        "--pick-ball",
        action="store_true",
        help="Review the ball on the still: click to place or move it, Delete to remove it",
    )
    parser.add_argument(
        "--remove-detections",
        action="store_true",
        help="Review the detections on the still and click false ones to remove them",
    )
    args = parser.parse_args()

    # Load the pose model first: a missing file should stop the run before
    # any calibration or review work is done.
    pose_model = pose.load_model(args.pose_weights)
    detection_model = player_detection.load_model(args.player_weights)
    run(
        args.input_path,
        args.output_path,
        detection_model,
        args.weights_kp,
        args.weights_line,
        ball_review=ball_picker.review_ball if args.pick_ball else None,
        review=detection_picker.review_detections if args.remove_detections else None,
        pose_model=pose_model,
    )


if __name__ == "__main__":
    main()
