from PIL import Image, ImageDraw

from footsight.team_classification import (
    classify_players,
    team_kit_colors,
    OFFICIALS_LABEL,
    TEAM_A_LABEL,
    TEAM_B_LABEL,
)


def _make_test_image(path, boxes_and_colors):
    image = Image.new("RGB", (400, 400), color=(34, 139, 34))
    draw = ImageDraw.Draw(image)
    for box, color in boxes_and_colors:
        draw.rectangle(list(box), fill=color)
    image.save(path)


def test_classify_players_splits_two_teams(tmp_path):
    image_path = tmp_path / "still.png"

    team_a_boxes = [(10.0, 10.0, 30.0, 70.0), (50.0, 10.0, 70.0, 70.0), (90.0, 10.0, 110.0, 70.0)]
    team_b_boxes = [(150.0, 10.0, 170.0, 70.0), (190.0, 10.0, 210.0, 70.0), (230.0, 10.0, 250.0, 70.0)]

    boxes_and_colors = (
        [(b, (0, 0, 220)) for b in team_a_boxes]  # blue jerseys
        + [(b, (220, 0, 0)) for b in team_b_boxes]  # red jerseys
    )
    _make_test_image(image_path, boxes_and_colors)

    all_boxes = team_a_boxes + team_b_boxes
    labels = classify_players(str(image_path), all_boxes)

    assert len(labels) == len(all_boxes)
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[3]
    assert {labels[0], labels[3]} == {TEAM_A_LABEL, TEAM_B_LABEL}


def test_classify_players_labels_an_odd_hue_as_officials_without_merging_the_teams(tmp_path):
    """A referee the detector mislabeled as a player wears a hue far from
    both teams. Left in, that outlier takes one of the two clusters and both
    real teams collapse into the other -- seen on the real Barca vs Feyenoord
    still, where the yellow referee (hue ~34) sat 65 hue units from the
    nearest team while the teams themselves were ~15 apart. The outlier is
    trimmed before the team centers are fitted, and labeled officials.
    """
    image_path = tmp_path / "still.png"

    team_a_boxes = [(10.0, 10.0, 30.0, 70.0), (50.0, 10.0, 70.0, 70.0), (90.0, 10.0, 110.0, 70.0)]
    team_b_boxes = [(150.0, 10.0, 170.0, 70.0), (190.0, 10.0, 210.0, 70.0), (230.0, 10.0, 250.0, 70.0)]
    odd_box = (300.0, 10.0, 320.0, 70.0)

    boxes_and_colors = (
        [(b, (0, 0, 220)) for b in team_a_boxes]  # blue, hue ~120
        + [(b, (0, 190, 220)) for b in team_b_boxes]  # light blue, hue ~98
        + [(odd_box, (220, 220, 0)) ]  # yellow -- far from both, like a referee
    )
    _make_test_image(image_path, boxes_and_colors)

    labels = classify_players(str(image_path), team_a_boxes + team_b_boxes + [odd_box])

    assert labels[6] == OFFICIALS_LABEL
    assert len(set(labels[:3])) == 1, f"team A did not hold together: {labels}"
    assert len(set(labels[3:6])) == 1, f"team B did not hold together: {labels}"
    assert labels[0] != labels[3]
    assert {labels[0], labels[3]} == {TEAM_A_LABEL, TEAM_B_LABEL}


def test_classify_players_ignores_grass_in_a_stretched_players_torso_sample(tmp_path):
    """A player with arms wide (or diving) fills only part of the torso
    window, so the sample is mostly pitch. Seen on the real 12.00.14 Barca
    still: two Barca players read as hue ~42 -- grass -- and were trimmed as
    officials. Grass-hued pixels are dropped before the median is taken,
    which puts them back with their team while a real referee's yellow is
    still trimmed.
    """
    image_path = tmp_path / "still.png"

    team_a_boxes = [(10.0, 10.0, 30.0, 70.0), (50.0, 10.0, 70.0, 70.0), (90.0, 10.0, 110.0, 70.0)]
    team_b_boxes = [(150.0, 10.0, 170.0, 70.0), (190.0, 10.0, 210.0, 70.0), (230.0, 10.0, 250.0, 70.0)]
    referee_box = (270.0, 10.0, 290.0, 70.0)
    # Detected box is 40px wide, but the blue jersey only covers a 6px strip
    # in the middle -- ~30% of the torso window; the rest is pitch.
    stretched_box = (310.0, 10.0, 350.0, 70.0)
    stretched_jersey = (327.0, 10.0, 333.0, 70.0)

    boxes_and_colors = (
        [(b, (0, 0, 220)) for b in team_a_boxes]  # blue, hue ~120
        + [(b, (0, 190, 220)) for b in team_b_boxes]  # light blue, hue ~98
        + [(referee_box, (220, 220, 0))]  # yellow referee, hue ~30
        + [(stretched_jersey, (0, 0, 220))]  # team A player, mostly grass
    )
    _make_test_image(image_path, boxes_and_colors)

    labels = classify_players(str(image_path), team_a_boxes + team_b_boxes + [referee_box, stretched_box])

    assert labels[7] == labels[0], f"stretched player not grouped with team A: {labels}"
    assert labels[6] == OFFICIALS_LABEL
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:6])) == 1
    assert labels[0] != labels[3]


def test_classify_players_keeps_a_striped_kit_across_the_red_wraparound_with_its_team(tmp_path):
    """Hue is a circle: 179 and 0 are neighbors. A kit striped in two reds
    either side of that seam (here hue ~175 and ~3) has a straight-line
    median of ~89 -- a color the jersey doesn't contain -- which lands it
    with the other team. Seen on the real 11.59.40 still, where a Barca
    player split 50/50 between claret (wrapping past 0) and blue read 75
    and was put with Feyenoord.
    """
    image_path = tmp_path / "still.png"

    team_a_boxes = [(10.0, 10.0, 30.0, 70.0), (50.0, 10.0, 70.0, 70.0), (90.0, 10.0, 110.0, 70.0)]
    team_b_boxes = [(150.0, 10.0, 170.0, 70.0), (190.0, 10.0, 210.0, 70.0), (230.0, 10.0, 250.0, 70.0)]
    striped_box = (270.0, 10.0, 290.0, 70.0)

    boxes_and_colors = (
        [(b, (220, 0, 37)) for b in team_a_boxes]  # crimson, hue ~175
        + [(b, (0, 147, 220)) for b in team_b_boxes]  # sky blue, hue ~100
        + [(striped_box, (220, 22, 0))]  # red-orange base, hue ~3 ...
        # ... with crimson stripes on every other column of the torso window
        + [((x, 10.0, x, 70.0), (220, 0, 37)) for x in range(275, 285, 2)]
    )
    _make_test_image(image_path, boxes_and_colors)

    labels = classify_players(str(image_path), team_a_boxes + team_b_boxes + [striped_box])

    assert labels[6] == labels[0], f"striped player not grouped with team A: {labels}"
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:6])) == 1
    assert labels[0] != labels[3]


def test_classify_players_labels_nobody_officials_when_only_two_teams_are_present(tmp_path):
    """Trimming must not invent an officials group out of ordinary
    within-team color spread: on the real Bayern vs Bodo/Glimt still every
    detection is an outfield player and none should be trimmed."""
    image_path = tmp_path / "still.png"

    team_a_boxes = [(10.0, 10.0, 30.0, 70.0), (50.0, 10.0, 70.0, 70.0), (90.0, 10.0, 110.0, 70.0)]
    team_b_boxes = [(150.0, 10.0, 170.0, 70.0), (190.0, 10.0, 210.0, 70.0), (230.0, 10.0, 250.0, 70.0)]

    boxes_and_colors = (
        [(b, (220, 20, 20)) for b in team_a_boxes]  # red, hue ~0
        + [(b, (220, 200, 20)) for b in team_b_boxes]  # yellow, hue ~27
    )
    _make_test_image(image_path, boxes_and_colors)

    labels = classify_players(str(image_path), team_a_boxes + team_b_boxes)

    assert OFFICIALS_LABEL not in labels
    assert labels[0] != labels[3]


def test_classify_players_resists_shadow_on_one_player(tmp_path):
    """A player in shadow has the same jersey hue as their teammates but much
    lower brightness -- this must not split them off from their team. This is
    a forward-looking spec/regression guard for the intended robustness
    property, not a reproduction of the real misclassification: that came
    from noise in real photographic crops averaged over a rectangle, which a
    flat-color synthetic rectangle can't recreate. The real fix is verified
    against the real sample still separately."""
    image_path = tmp_path / "still.png"

    team_a_boxes = [(10.0, 10.0, 30.0, 70.0), (50.0, 10.0, 70.0, 70.0), (90.0, 10.0, 110.0, 70.0)]
    team_b_boxes = [(150.0, 10.0, 170.0, 70.0), (190.0, 10.0, 210.0, 70.0), (230.0, 10.0, 250.0, 70.0)]

    # Team A's jerseys are pure blue; the last one is the same hue but much
    # darker, as if standing in shadow.
    boxes_and_colors = (
        [(team_a_boxes[0], (0, 0, 220)), (team_a_boxes[1], (0, 0, 220)), (team_a_boxes[2], (0, 0, 60))]
        + [(b, (220, 0, 0)) for b in team_b_boxes]  # red jerseys
    )
    _make_test_image(image_path, boxes_and_colors)

    labels = classify_players(str(image_path), team_a_boxes + team_b_boxes)

    assert len(set(labels[:3])) == 1, f"shadowed player split from teammates: {labels}"
    assert labels[0] != labels[3]


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

    # Hue ~5 (red) vs hue ~20 (orange) -- a close hue gap. Within each team,
    # one player is lower-saturation than the other two (same hue).
    boxes_and_colors = [
        (team_a_boxes[0], (220, 62, 30)),  # team A, high saturation
        (team_a_boxes[1], (220, 62, 30)),  # team A, high saturation
        (team_a_boxes[2], (220, 134, 116)),  # team A, low saturation
        (team_b_boxes[0], (220, 157, 30)),  # team B, high saturation
        (team_b_boxes[1], (220, 157, 30)),  # team B, high saturation
        (team_b_boxes[2], (220, 185, 116)),  # team B, low saturation
    ]
    _make_test_image(image_path, boxes_and_colors)

    labels = classify_players(str(image_path), team_a_boxes + team_b_boxes)

    assert len(set(labels[:3])) == 1, f"team A split by saturation instead of staying grouped by hue: {labels}"
    assert len(set(labels[3:])) == 1, f"team B split by saturation instead of staying grouped by hue: {labels}"
    assert labels[0] != labels[3]


def test_classify_players_returns_default_label_when_too_few_players(tmp_path):
    image_path = tmp_path / "still.png"
    _make_test_image(image_path, [((10.0, 10.0, 30.0, 70.0), (0, 0, 220))])

    assert classify_players(str(image_path), [(10.0, 10.0, 30.0, 70.0)]) == [TEAM_A_LABEL]


def test_classify_players_handles_no_players(tmp_path):
    image_path = tmp_path / "still.png"
    _make_test_image(image_path, [])

    assert classify_players(str(image_path), []) == []


def _draw_kit(boxes, shirt, shorts):
    """Shirt over the top 48% of each box, shorts over the next 17% --
    covering where the shirt and shorts bands are sampled."""
    drawn = []
    for x1, y1, x2, y2 in boxes:
        height = y2 - y1
        drawn.append(((x1, y1, x2, y1 + height * 0.48), shirt))
        drawn.append(((x1, y1 + height * 0.48, x2, y1 + height * 0.65), shorts))
    return drawn


def _close(color, expected, tolerance=12):
    return all(abs(a - b) <= tolerance for a, b in zip(color, expected))


def test_team_kit_colors_samples_each_teams_shirt_and_shorts(tmp_path):
    """The mockup draws each team in the kit it actually wears (orange
    Netherlands, not a fixed blue), so the shirt and shorts colors are read
    off the still."""
    image_path = tmp_path / "still.png"
    team_a_boxes = [(10.0, 10.0, 30.0, 70.0), (50.0, 10.0, 70.0, 70.0)]
    team_b_boxes = [(150.0, 10.0, 170.0, 70.0), (190.0, 10.0, 210.0, 70.0)]
    _make_test_image(
        image_path,
        _draw_kit(team_a_boxes, (245, 130, 30), (240, 240, 240))  # orange shirt, white shorts
        + _draw_kit(team_b_boxes, (30, 30, 35), (200, 20, 40)),  # black shirt, red shorts
    )

    colors = team_kit_colors(
        str(image_path),
        team_a_boxes + team_b_boxes,
        [TEAM_A_LABEL, TEAM_A_LABEL, TEAM_B_LABEL, TEAM_B_LABEL],
    )

    shirt_a, shorts_a = colors[TEAM_A_LABEL]
    shirt_b, shorts_b = colors[TEAM_B_LABEL]
    assert _close(shirt_a, (245, 130, 30)), shirt_a
    assert _close(shorts_a, (240, 240, 240)), shorts_a
    assert _close(shirt_b, (30, 30, 35)), shirt_b
    assert _close(shorts_b, (200, 20, 40)), shorts_b


def test_team_kit_colors_ignores_officials(tmp_path):
    image_path = tmp_path / "still.png"
    boxes = [(10.0, 10.0, 30.0, 70.0), (150.0, 10.0, 170.0, 70.0), (300.0, 10.0, 320.0, 70.0)]
    _make_test_image(
        image_path,
        _draw_kit([boxes[0]], (245, 130, 30), (240, 240, 240))
        + _draw_kit([boxes[1]], (30, 30, 35), (30, 30, 35))
        + _draw_kit([boxes[2]], (230, 230, 0), (20, 20, 20)),  # referee
    )

    colors = team_kit_colors(str(image_path), boxes, [TEAM_A_LABEL, TEAM_B_LABEL, OFFICIALS_LABEL])

    assert set(colors) == {TEAM_A_LABEL, TEAM_B_LABEL}


def test_team_kit_colors_gives_up_when_the_two_kits_would_look_alike(tmp_path):
    """Two dark kits that the jersey hue can still split may look the same
    once drawn; then the mockup falls back to its fixed, clearly different
    team colors rather than showing two near-identical teams."""
    image_path = tmp_path / "still.png"
    boxes = [(10.0, 10.0, 30.0, 70.0), (150.0, 10.0, 170.0, 70.0)]
    _make_test_image(
        image_path,
        _draw_kit([boxes[0]], (35, 30, 30), (20, 20, 20))
        + _draw_kit([boxes[1]], (30, 30, 45), (20, 20, 20)),
    )

    assert team_kit_colors(str(image_path), boxes, [TEAM_A_LABEL, TEAM_B_LABEL]) == {}


def test_team_kit_colors_gives_up_when_a_team_has_no_players(tmp_path):
    image_path = tmp_path / "still.png"
    boxes = [(10.0, 10.0, 30.0, 70.0)]
    _make_test_image(image_path, _draw_kit(boxes, (245, 130, 30), (240, 240, 240)))

    assert team_kit_colors(str(image_path), boxes, [TEAM_A_LABEL]) == {}
