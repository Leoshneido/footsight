import torch
from PIL import Image

from footsight.player_detection import detect_players, PERSON_LABEL


class FakeModel:
    def __init__(self, boxes, labels, scores):
        self._boxes = boxes
        self._labels = labels
        self._scores = scores

    def __call__(self, tensor):
        return [{
            "boxes": torch.tensor(self._boxes, dtype=torch.float32),
            "labels": torch.tensor(self._labels, dtype=torch.int64),
            "scores": torch.tensor(self._scores, dtype=torch.float32),
        }]


def _make_test_image(path):
    Image.new("RGB", (100, 100), color=(0, 128, 0)).save(path)


def test_detect_players_filters_to_person_label_above_confidence(tmp_path):
    image_path = tmp_path / "still.png"
    _make_test_image(image_path)

    model = FakeModel(
        boxes=[[10.0, 20.0, 30.0, 60.0], [5.0, 5.0, 15.0, 15.0], [40.0, 40.0, 50.0, 50.0]],
        labels=[PERSON_LABEL, 3, PERSON_LABEL],
        scores=[0.9, 0.95, 0.2],
    )

    boxes = detect_players(str(image_path), model, confidence=0.5)

    assert boxes == [(10.0, 20.0, 30.0, 60.0)]


def test_detect_players_returns_empty_list_when_no_people(tmp_path):
    image_path = tmp_path / "still.png"
    _make_test_image(image_path)

    model = FakeModel(boxes=[], labels=[], scores=[])

    boxes = detect_players(str(image_path), model)

    assert boxes == []
