from PIL import Image

from footsight.player_detection import (
    detect_players,
    GOALKEEPER_ROLE,
    PLAYER_ROLE,
    REFEREE_ROLE,
)


class FakeBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = [xyxy]
        self.cls = cls
        self.conf = conf


class FakeResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


class FakeModel:
    """Stands in for an ultralytics YOLO model, whose predict() returns one
    Results object per source image."""

    NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

    def __init__(self, boxes):
        self._boxes = boxes
        self.predict_kwargs = None

    def predict(self, source, **kwargs):
        self.predict_kwargs = kwargs
        return [FakeResult(self.NAMES, self._boxes)]


def _make_test_image(path):
    Image.new("RGB", (100, 100), color=(0, 128, 0)).save(path)


def test_detect_players_returns_each_detection_with_its_role(tmp_path):
    image_path = tmp_path / "still.png"
    _make_test_image(image_path)

    model = FakeModel([
        FakeBox([10.0, 20.0, 30.0, 60.0], cls=2, conf=0.9),
        FakeBox([40.0, 20.0, 60.0, 60.0], cls=1, conf=0.95),
        FakeBox([70.0, 20.0, 90.0, 60.0], cls=3, conf=0.8),
    ])

    detections = detect_players(str(image_path), model)

    assert detections == [
        ((10.0, 20.0, 30.0, 60.0), PLAYER_ROLE),
        ((40.0, 20.0, 60.0, 60.0), GOALKEEPER_ROLE),
        ((70.0, 20.0, 90.0, 60.0), REFEREE_ROLE),
    ]


def test_detect_players_drops_the_models_own_ball_class(tmp_path):
    """The player model also emits a "ball" class, but footsight finds the
    ball separately -- a ball box is not a person and must not be projected
    as one."""
    image_path = tmp_path / "still.png"
    _make_test_image(image_path)

    model = FakeModel([
        FakeBox([10.0, 20.0, 30.0, 60.0], cls=2, conf=0.9),
        FakeBox([50.0, 50.0, 56.0, 56.0], cls=0, conf=0.9),
    ])

    detections = detect_players(str(image_path), model)

    assert detections == [((10.0, 20.0, 30.0, 60.0), PLAYER_ROLE)]


def test_detect_players_drops_detections_below_confidence(tmp_path):
    """Low-confidence boxes are broadcast furniture, not people -- on a real
    still the watermark scored 0.65 while every real player scored 0.86+."""
    image_path = tmp_path / "still.png"
    _make_test_image(image_path)

    model = FakeModel([
        FakeBox([10.0, 20.0, 30.0, 60.0], cls=2, conf=0.9),
        FakeBox([70.0, 70.0, 80.0, 80.0], cls=2, conf=0.65),
    ])

    detections = detect_players(str(image_path), model, confidence=0.7)

    assert detections == [((10.0, 20.0, 30.0, 60.0), PLAYER_ROLE)]


def test_detect_players_returns_empty_list_when_nothing_detected(tmp_path):
    image_path = tmp_path / "still.png"
    _make_test_image(image_path)

    assert detect_players(str(image_path), FakeModel([])) == []
