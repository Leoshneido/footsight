from PIL import Image, ImageDraw

from footsight.team_classification import classify_players, TEAM_A_LABEL, TEAM_B_LABEL, OFFICIALS_LABEL


def _make_test_image(path, boxes_and_colors):
    image = Image.new("RGB", (400, 400), color=(34, 139, 34))
    draw = ImageDraw.Draw(image)
    for box, color in boxes_and_colors:
        draw.rectangle(list(box), fill=color)
    image.save(path)


def test_classify_players_splits_two_teams_and_a_referee(tmp_path):
    image_path = tmp_path / "still.png"

    team_a_boxes = [(10.0, 10.0, 30.0, 70.0), (50.0, 10.0, 70.0, 70.0), (90.0, 10.0, 110.0, 70.0)]
    team_b_boxes = [(150.0, 10.0, 170.0, 70.0), (190.0, 10.0, 210.0, 70.0), (230.0, 10.0, 250.0, 70.0)]
    referee_box = (300.0, 10.0, 320.0, 70.0)

    boxes_and_colors = (
        [(b, (0, 0, 220)) for b in team_a_boxes]  # blue jerseys
        + [(b, (220, 0, 0)) for b in team_b_boxes]  # red jerseys
        + [(referee_box, (0, 220, 220))]  # cyan -- distinct from both teams
    )
    _make_test_image(image_path, boxes_and_colors)

    all_boxes = team_a_boxes + team_b_boxes + [referee_box]
    labels = classify_players(str(image_path), all_boxes)

    assert len(labels) == len(all_boxes)
    team_labels = labels[:3]
    other_team_labels = labels[3:6]
    referee_label = labels[6]

    assert len(set(team_labels)) == 1
    assert len(set(other_team_labels)) == 1
    assert team_labels[0] != other_team_labels[0]
    assert referee_label == OFFICIALS_LABEL
    assert {team_labels[0], other_team_labels[0]} == {TEAM_A_LABEL, TEAM_B_LABEL}


def test_classify_players_returns_default_label_when_too_few_players(tmp_path):
    image_path = tmp_path / "still.png"
    _make_test_image(image_path, [((10.0, 10.0, 30.0, 70.0), (0, 0, 220))])

    labels = classify_players(str(image_path), [(10.0, 10.0, 30.0, 70.0)])

    assert labels == [TEAM_A_LABEL]


def test_classify_players_returns_empty_list_for_no_players(tmp_path):
    image_path = tmp_path / "still.png"
    Image.new("RGB", (400, 400), color=(34, 139, 34)).save(image_path)

    assert classify_players(str(image_path), []) == []
