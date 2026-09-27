from footsight.detection_picker import detection_at, drop_detections

PLAYER = "player"


def test_detection_at_returns_the_box_under_the_click():
    detections = [((0.0, 0.0, 10.0, 20.0), PLAYER), ((30.0, 0.0, 40.0, 20.0), PLAYER)]

    assert detection_at((35.0, 10.0), detections) == 1


def test_detection_at_returns_none_when_the_click_misses_every_box():
    detections = [((0.0, 0.0, 10.0, 20.0), PLAYER)]

    assert detection_at((50.0, 50.0), detections) is None


def test_detection_at_picks_the_smallest_box_where_boxes_overlap():
    """Overlapping players (one partly behind another) produce nested or
    overlapping boxes; the smaller one is the one the click is aimed at --
    the bigger one can still be picked by clicking outside the overlap."""
    detections = [((0.0, 0.0, 40.0, 80.0), PLAYER), ((10.0, 10.0, 25.0, 40.0), PLAYER)]

    assert detection_at((15.0, 20.0), detections) == 1


def test_drop_detections_removes_only_the_chosen_ones():
    detections = [
        ((0.0, 0.0, 10.0, 20.0), PLAYER),
        ((30.0, 0.0, 40.0, 20.0), PLAYER),
        ((60.0, 0.0, 70.0, 20.0), "referee"),
    ]

    assert drop_detections(detections, {1}) == [detections[0], detections[2]]
