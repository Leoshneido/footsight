"""Per-player pose estimation for the camera-angle view.

Players in broadcast stills are only ~70-120 px tall, so each one is cropped
with some padding and enlarged before the pose model sees it; the joints
found are mapped back to still pixels. A pose that can't be trusted comes
back as None, and the player is drawn as the standing figure instead.
"""
from pathlib import Path

import cv2
import numpy as np

DEFAULT_POSE_WEIGHTS = Path("weights/ultralytics/yolo11m-pose.pt")

# Crop padding around the detected box, as fractions of its width/height.
CROP_PAD_SIDE = 0.35
CROP_PAD_TOP = 0.20
CROP_PAD_BOTTOM = 0.15
CROP_HEIGHT_PX = 320
POSE_CONFIDENCE = 0.1

# When a pose is trusted (see is_reliable).
JOINT_CONFIDENCE = 0.3
MIN_CONFIDENT_JOINTS = 10
MAX_OVERLAP_IOU = 0.2
SKELETON_HEIGHT_RANGE = (0.6, 1.4)

HEAD_JOINTS = range(0, 5)
SHOULDERS = (5, 6)
ANKLES = (15, 16)


def load_model(weights_path: str | Path = DEFAULT_POSE_WEIGHTS):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Pose model not found at {weights_path} -- see the pose model download step in README.md"
        )
    from ultralytics import YOLO

    return YOLO(str(weights_path))


def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def crop_window(box, image_w: int, image_h: int) -> tuple[int, int, int, int]:
    """The padded crop around a player's box, clamped to the image."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (
        int(max(0, x1 - CROP_PAD_SIDE * w)),
        int(max(0, y1 - CROP_PAD_TOP * h)),
        int(min(image_w, x2 + CROP_PAD_SIDE * w)),
        int(min(image_h, y2 + CROP_PAD_BOTTOM * h)),
    )


def pick_person(person_boxes, target_box) -> int | None:
    """Which of the people the model found in a crop is the player we cropped
    for: the one whose box overlaps theirs most."""
    overlaps = [box_iou(box, target_box) for box in person_boxes]
    if not overlaps or max(overlaps) == 0:
        return None
    return int(np.argmax(overlaps))


def is_reliable(pose: np.ndarray, box, other_boxes) -> bool:
    """Whether a pose is trustworthy enough to draw. Rejected: too few
    confident joints; a player tangled with another (the model can't tell
    whose limbs are whose); a skeleton whose height doesn't fit the box,
    which catches one merged across two people."""
    confident = pose[:, 2] > JOINT_CONFIDENCE
    if confident.sum() < MIN_CONFIDENT_JOINTS:
        return False
    if any(box_iou(box, other) > MAX_OVERLAP_IOU for other in other_boxes):
        return False

    top_joints = [i for i in HEAD_JOINTS if confident[i]] or [i for i in SHOULDERS if confident[i]]
    bottom_joints = [i for i in ANKLES if confident[i]]
    if not top_joints or not bottom_joints:
        return False
    skeleton_height = pose[bottom_joints, 1].max() - pose[top_joints, 1].min()
    low, high = SKELETON_HEIGHT_RANGE
    return low <= skeleton_height / (box[3] - box[1]) <= high


def estimate_poses(image_path: str, boxes, model) -> list[np.ndarray | None]:
    """One pose per box: a (17, 3) array of joint x, y in still pixels and
    confidence (COCO keypoint order), or None when it can't be trusted."""
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Could not read {image_path} for pose estimation")
    image_h, image_w = image.shape[:2]

    poses = []
    for index, box in enumerate(boxes):
        cx1, cy1, cx2, cy2 = crop_window(box, image_w, image_h)
        crop = image[cy1:cy2, cx1:cx2]
        scale = CROP_HEIGHT_PX / crop.shape[0]
        enlarged = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), CROP_HEIGHT_PX), interpolation=cv2.INTER_CUBIC)
        scale_x = enlarged.shape[1] / crop.shape[1]
        result = model.predict(enlarged, conf=POSE_CONFIDENCE, imgsz=CROP_HEIGHT_PX, verbose=False)[0]

        if result.keypoints is None or len(result.boxes) == 0:
            poses.append(None)
            continue
        target = ((box[0] - cx1) * scale_x, (box[1] - cy1) * scale, (box[2] - cx1) * scale_x, (box[3] - cy1) * scale)
        chosen = pick_person(result.boxes.xyxy.cpu().numpy(), target)
        if chosen is None:
            poses.append(None)
            continue

        xy = result.keypoints.xy.cpu().numpy()[chosen]
        conf = result.keypoints.conf.cpu().numpy()[chosen]
        pose = np.column_stack([xy[:, 0] / scale_x + cx1, xy[:, 1] / scale + cy1, conf])
        others = [other for j, other in enumerate(boxes) if j != index]
        poses.append(pose if is_reliable(pose, box, others) else None)
    return poses
