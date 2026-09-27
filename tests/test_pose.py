import numpy as np
import pytest
from PIL import Image

from footsight.pose import box_iou, crop_window, estimate_poses, is_reliable, pick_person

# COCO order: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows,
# 9-10 wrists, 11-12 hips, 13-14 knees, 15-16 ankles.
STANDING = np.array([
    (0.50, 0.00), (0.48, -0.02), (0.52, -0.02), (0.45, 0.00), (0.55, 0.00),
    (0.35, 0.18), (0.65, 0.18), (0.30, 0.38), (0.70, 0.38), (0.28, 0.55), (0.72, 0.55),
    (0.40, 0.52), (0.60, 0.52), (0.40, 0.76), (0.60, 0.76), (0.40, 1.00), (0.60, 1.00),
])


def _pose_in_box(box, conf=0.9, height_fraction=0.9):
    """A clean standing skeleton filling height_fraction of the box."""
    x1, y1, x2, y2 = box
    h = (y2 - y1) * height_fraction
    top = y1 + (y2 - y1 - h) / 2
    xs = x1 + STANDING[:, 0] * (x2 - x1)
    ys = top + STANDING[:, 1] * h
    return np.column_stack([xs, ys, np.full(17, conf)])


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_crop_window_pads_the_box_and_stays_inside_the_image():
    assert crop_window((100, 100, 120, 160), 1000, 1000) == (93, 88, 127, 169)
    assert crop_window((0, 0, 20, 60), 1000, 1000)[:2] == (0, 0)
    assert crop_window((980, 940, 1000, 1000), 1000, 1000)[2:] == (1000, 1000)


def test_pick_person_takes_the_best_overlap():
    target = (40, 40, 60, 100)
    assert pick_person([(0, 0, 20, 30), (38, 42, 62, 98)], target) == 1
    assert pick_person([], target) is None
    assert pick_person([(0, 0, 5, 5)], target) is None


def test_is_reliable_accepts_a_clean_pose():
    box = (100, 100, 140, 200)
    assert is_reliable(_pose_in_box(box), box, [(500, 500, 540, 600)])


def test_is_reliable_rejects_too_few_confident_joints():
    box = (100, 100, 140, 200)
    pose = _pose_in_box(box)
    pose[9:, 2] = 0.1  # only 9 joints confident
    assert not is_reliable(pose, box, [])


def test_is_reliable_rejects_players_tangled_with_another():
    """Two overlapping players: the model can't tell whose limbs are whose."""
    box = (100, 100, 140, 200)
    assert not is_reliable(_pose_in_box(box), box, [(115, 100, 155, 200)])


def test_is_reliable_rejects_a_skeleton_the_wrong_size_for_its_box():
    """A skeleton half the box's height -- or one merged across two
    people -- isn't this player's pose."""
    box = (100, 100, 140, 200)
    assert not is_reliable(_pose_in_box(box, height_fraction=0.5), box, [])


class _Arr:
    def __init__(self, a):
        self._a = np.asarray(a, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    def __len__(self):
        return len(self._a)


class _Keypoints:
    def __init__(self, xy, conf):
        self.xy, self.conf = _Arr(xy), _Arr(conf)


class _Boxes:
    def __init__(self, xyxy):
        self.xyxy = _Arr(xyxy)

    def __len__(self):
        return len(self.xyxy)


class _Result:
    def __init__(self, boxes, xy, conf):
        self.boxes = _Boxes(boxes)
        self.keypoints = _Keypoints(xy, conf) if len(boxes) else None


class FakePoseModel:
    """Stands in for an ultralytics pose model: finds one person filling the
    crop, with a standing skeleton placed by fractions of the crop size."""

    def __init__(self, fractions=None):
        self.fractions = fractions
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append((image.shape, kwargs))
        h, w = image.shape[:2]
        if self.fractions is None:
            return [_Result(np.zeros((0, 4)), np.zeros((0, 17, 2)), np.zeros((0, 17)))]
        xy = self.fractions * np.array([w, h])
        return [_Result(np.array([[0, 0, w, h]]), xy[None], np.full((1, 17), 0.9))]


def test_estimate_poses_maps_crop_joints_back_to_the_still(tmp_path):
    image_path = tmp_path / "still.png"
    Image.new("RGB", (1000, 1000), (40, 120, 40)).save(image_path)
    box = (100, 100, 120, 160)
    cx1, cy1, cx2, cy2 = crop_window(box, 1000, 1000)
    # skeleton from 18% to 88% of the crop height, centered
    fractions = np.column_stack([0.3 + STANDING[:, 0] * 0.4, 0.18 + STANDING[:, 1] * 0.70])
    model = FakePoseModel(fractions)

    [pose] = estimate_poses(str(image_path), [box], model)

    expected_x = cx1 + fractions[:, 0] * (cx2 - cx1)
    expected_y = cy1 + fractions[:, 1] * (cy2 - cy1)
    assert pose[:, 0] == pytest.approx(expected_x, abs=0.6)
    assert pose[:, 1] == pytest.approx(expected_y, abs=0.6)
    shape, kwargs = model.calls[0]
    assert shape[0] == 320  # crop enlarged to 320 px tall
    assert kwargs["conf"] == 0.1 and kwargs["imgsz"] == 320


def test_estimate_poses_returns_none_when_nobody_is_found(tmp_path):
    image_path = tmp_path / "still.png"
    Image.new("RGB", (1000, 1000), (40, 120, 40)).save(image_path)

    assert estimate_poses(str(image_path), [(100, 100, 120, 160)], FakePoseModel(None)) == [None]


def test_load_model_explains_a_missing_weights_file(tmp_path):
    from footsight.pose import load_model

    with pytest.raises(FileNotFoundError, match="README"):
        load_model(tmp_path / "missing.pt")
