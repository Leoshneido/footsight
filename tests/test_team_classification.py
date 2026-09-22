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


def test_classify_players_resists_shadow_on_one_player(tmp_path):
    """A player in shadow has the same jersey hue/saturation as their
    teammates but much lower brightness -- this must not push them into
    the officials cluster. This is a forward-looking spec/regression guard
    for the intended robustness property, not a reproduction of the real
    22-player misclassification: that came from noise in real photographic
    crops averaged over a rectangle, which a flat-color synthetic rectangle
    can't recreate. The real fix is verified against the real sample still
    separately."""
    image_path = tmp_path / "still.png"

    team_a_boxes = [(10.0, 10.0, 30.0, 70.0), (50.0, 10.0, 70.0, 70.0), (90.0, 10.0, 110.0, 70.0)]
    team_b_boxes = [(150.0, 10.0, 170.0, 70.0), (190.0, 10.0, 210.0, 70.0), (230.0, 10.0, 250.0, 70.0)]
    referee_box = (300.0, 10.0, 320.0, 70.0)

    # Team A's jerseys are pure blue; the last one is the same hue/saturation
    # but much darker, as if standing in shadow.
    boxes_and_colors = (
        [(team_a_boxes[0], (0, 0, 220)), (team_a_boxes[1], (0, 0, 220)), (team_a_boxes[2], (0, 0, 60))]
        + [(b, (220, 0, 0)) for b in team_b_boxes]  # red jerseys
        + [(referee_box, (0, 220, 220))]  # cyan -- distinct from both teams
    )
    _make_test_image(image_path, boxes_and_colors)

    all_boxes = team_a_boxes + team_b_boxes + [referee_box]
    labels = classify_players(str(image_path), all_boxes)

    assert len(set(labels[:3])) == 1, f"shadowed player split from teammates: {labels}"
    assert labels[:3][0] != OFFICIALS_LABEL
    assert labels[6] == OFFICIALS_LABEL


def test_classify_players_separates_close_hues_despite_saturation_spread(tmp_path):
    """Two teams with close-but-distinct hues (red vs orange) and realistic
    within-team saturation spread (lighting/motion blur) must still split by
    hue, not by saturation. Saturation's numeric range (0-255) is much wider
    than hue's (0-179), so clustering on raw (H, S) lets within-team
    saturation noise swamp the actual between-team hue signal -- confirmed
    on a real Bayern (red) vs Bodo/Glimt (yellow) still, where every player
    from both teams landed in one cluster instead of splitting.
    """
    image_path = tmp_path / "still.png"

    team_a_boxes = [(10.0, 10.0, 30.0, 70.0), (50.0, 10.0, 70.0, 70.0), (90.0, 10.0, 110.0, 70.0)]
    team_b_boxes = [(150.0, 10.0, 170.0, 70.0), (190.0, 10.0, 210.0, 70.0), (230.0, 10.0, 250.0, 70.0)]
    referee_box = (300.0, 10.0, 320.0, 70.0)

    # Hue ~5 (red) vs hue ~20 (orange) -- a close hue gap. Within each team,
    # one player is lower-saturation than the other two (same hue), same
    # shape of noise the shadow test guards against but combined with a
    # close hue gap instead of a far one.
    boxes_and_colors = [
        (team_a_boxes[0], (220, 62, 30)),  # team A, high saturation
        (team_a_boxes[1], (220, 62, 30)),  # team A, high saturation
        (team_a_boxes[2], (220, 134, 116)),  # team A, low saturation
        (team_b_boxes[0], (220, 157, 30)),  # team B, high saturation
        (team_b_boxes[1], (220, 157, 30)),  # team B, high saturation
        (team_b_boxes[2], (220, 185, 116)),  # team B, low saturation
        (referee_box, (168, 220, 220)),  # distinct hue, low saturation
    ]
    _make_test_image(image_path, boxes_and_colors)

    all_boxes = team_a_boxes + team_b_boxes + [referee_box]
    labels = classify_players(str(image_path), all_boxes)

    assert len(set(labels[:3])) == 1, f"team A split by saturation instead of staying grouped by hue: {labels}"
    assert len(set(labels[3:6])) == 1, f"team B split by saturation instead of staying grouped by hue: {labels}"
    assert labels[0] != labels[3]
    assert labels[6] == OFFICIALS_LABEL


def test_classify_players_returns_default_label_when_too_few_players(tmp_path):
    image_path = tmp_path / "still.png"
    _make_test_image(image_path, [((10.0, 10.0, 30.0, 70.0), (0, 0, 220))])

    labels = classify_players(str(image_path), [(10.0, 10.0, 30.0, 70.0)])

    assert labels == [TEAM_A_LABEL]


def test_classify_players_returns_empty_list_for_no_players(tmp_path):
    image_path = tmp_path / "still.png"
    Image.new("RGB", (400, 400), color=(34, 139, 34)).save(image_path)

    assert classify_players(str(image_path), []) == []
