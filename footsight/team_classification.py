import cv2
import numpy as np
from PIL import Image

TEAM_A_LABEL = "team_a"
TEAM_B_LABEL = "team_b"
OFFICIALS_LABEL = "officials"

# How far outside the quartile spread of jersey hues a player has to sit to
# be treated as an odd one out rather than a teammate. 1.5 is the standard
# Tukey fence; on the sample stills it trims the referee's yellow (65 hue
# units from the nearest team) while leaving ordinary within-team spread
# (~15 units, and ~20 between the two Bayern/Bodo kits) untouched.
OUTLIER_IQR_MULTIPLIER = 1.5

# OpenCV hue band (0-179 scale) of the pitch. A stretched or diving player
# fills only part of the torso window, and without this the grass around
# them can outvote the jersey. The referee's yellow (~30-34) sits below it.
GRASS_HUE_RANGE = (40, 60)

# OpenCV stores hue as 0-179, so 180 wraps back to 0.
HUE_PERIOD = 180

# Where the drawn kit colors are sampled, as fractions of the box height.
# Higher than the jersey-hue torso window (0.3-0.6), which reaches into the
# shorts; measured on the sample stills, lower shorts bands mostly caught
# thighs and grass between the legs.
SHIRT_BAND = (0.25, 0.45)
SHORTS_BAND = (0.5, 0.62)
# Below this RGB distance two teams' shirts would look alike on the mockup.
KIT_MIN_DISTANCE = 60.0


def _jersey_hues(image_bgr: np.ndarray, player_box: tuple[float, float, float, float]) -> np.ndarray:
    """Hues of the pixels in a player's torso -- the middle third of their box
    vertically (avoiding the head and legs/socks) and the middle half
    horizontally (avoiding background/neighboring-player bleed at the box
    edges). Only hue is used: brightness (value) changes with shadow, and
    saturation can vary almost as much within one team's jersey (lighting,
    motion blur) as hue varies between two teams with close kit colors (e.g.
    red vs orange) -- letting saturation into the clustering feature let that
    noise swamp the real signal. Hue alone is the jersey's actual identity
    and is the axis teams/officials are chosen to be far apart on."""
    x1, y1, x2, y2 = (int(v) for v in player_box)
    height = y2 - y1
    width = x2 - x1
    torso_y1, torso_y2 = y1 + int(height * 0.3), y1 + int(height * 0.6)
    torso_x1, torso_x2 = x1 + int(width * 0.25), x1 + int(width * 0.75)

    torso = image_bgr[torso_y1:torso_y2, torso_x1:torso_x2]
    if torso.size == 0:
        torso = image_bgr[y1:y2, x1:x2]

    torso_hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hues = torso_hsv[..., 0].reshape(-1).astype(np.float32)
    # Drop the pitch showing through the crop. If nothing is left (a team
    # in a grass-green kit), keep every pixel rather than return nothing.
    low, high = GRASS_HUE_RANGE
    not_grass = hues[(hues < low) | (hues > high)]
    if not_grass.size > 0:
        hues = not_grass
    return hues


def _circular_median(hues: np.ndarray) -> float:
    """Median hue treating the scale as the circle it is (179 is next to
    0). The circle is cut at the widest empty stretch between the pixel
    hues, so a kit striped either side of the red seam reads as red rather
    than as the unrelated color halfway round."""
    distinct = np.unique(hues)
    gaps = np.diff(np.append(distinct, distinct[0] + HUE_PERIOD))
    cut = distinct[(int(np.argmax(gaps)) + 1) % len(distinct)]
    return float((np.median((hues - cut) % HUE_PERIOD) + cut) % HUE_PERIOD)


def _indices_within_hue_fence(hues: np.ndarray) -> list[int]:
    """Indices of the players whose jersey hue sits inside the Tukey fence
    of the whole set -- i.e. everyone except an odd color far from the bulk
    of the two teams."""
    first_quartile, third_quartile = np.percentile(hues, [25, 75])
    spread = third_quartile - first_quartile
    low = first_quartile - OUTLIER_IQR_MULTIPLIER * spread
    high = third_quartile + OUTLIER_IQR_MULTIPLIER * spread
    return [index for index, hue in enumerate(hues) if low <= hue <= high]


def classify_players(
    image_path: str,
    player_boxes: list[tuple[float, float, float, float]],
) -> list[str]:
    """Label each player "team_a", "team_b" or "officials" by clustering
    jersey hues into 2 team groups.

    Mostly outfield players reach this function -- the detector tags
    goalkeepers and officials with their own roles, and those are filtered
    out upstream -- which is what lets this fit exactly 2 clusters. But the
    detector does sometimes hand a referee over as a player, and a hue that
    far from both teams would take a whole cluster and merge the two real
    teams together. So hues outside the Tukey fence are trimmed before the
    team centers are fitted, and labeled officials instead.

    The trim and the team split use hue differently. The trim runs on the
    plain straight-line median, which keeps a referee's yellow clearly apart
    on every sample still. The team split runs on the circular median, so a
    striped kit either side of the red seam (hue ~179 and ~0) stays with its
    team -- on the real 11.59.40 still a Barca player half claret, half blue
    read 75 on the straight line and was put with Feyenoord. Doing the trim
    circularly too was tried: Barca's circular medians spread over 130-179,
    which widened the fence until the referee got through.

    Falls back to labeling everyone "team_a" when there are too few players
    to meaningfully form 2 clusters.
    """
    if len(player_boxes) < 2:
        return [TEAM_A_LABEL] * len(player_boxes)

    image = Image.open(image_path).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    pixel_hues = [_jersey_hues(image_bgr, box) for box in player_boxes]
    hues = np.array([np.median(player) for player in pixel_hues], dtype=np.float32)

    team_indices = _indices_within_hue_fence(hues)
    if len(team_indices) < 2:
        return [TEAM_A_LABEL] * len(player_boxes)

    # Cluster on the hue circle: each hue becomes a point on a unit circle,
    # so 179 and 0 sit next to each other.
    angles = np.array([_circular_median(pixel_hues[index]) for index in team_indices]) * (2 * np.pi / HUE_PERIOD)
    team_points = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    _, cluster_indices, _ = cv2.kmeans(team_points, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    cluster_indices = cluster_indices.flatten()

    first_cluster = int(cluster_indices[0])
    labels = [OFFICIALS_LABEL] * len(player_boxes)
    for position, index in enumerate(team_indices):
        cluster = cluster_indices[position]
        labels[index] = TEAM_A_LABEL if cluster == first_cluster else TEAM_B_LABEL
    return labels


def _band_color(
    image_bgr: np.ndarray,
    player_box: tuple[float, float, float, float],
    band: tuple[float, float],
) -> np.ndarray:
    """Median RGB color of a horizontal band of the box (middle half of its
    width), with pitch-green pixels left out."""
    x1, y1, x2, y2 = (int(v) for v in player_box)
    height, width = y2 - y1, x2 - x1
    top, bottom = y1 + int(height * band[0]), y1 + int(height * band[1])
    region = image_bgr[top:bottom, x1 + int(width * 0.25):x1 + int(width * 0.75)]
    if region.size == 0:
        region = image_bgr[y1:y2, x1:x2]

    pixels = region.reshape(-1, 3)
    hues = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)[..., 0].reshape(-1)
    low, high = GRASS_HUE_RANGE
    not_grass = pixels[(hues < low) | (hues > high)]
    if len(not_grass) > 0:
        pixels = not_grass
    blue, green, red = np.median(pixels, axis=0)
    return np.array([red, green, blue])


def team_kit_colors(
    image_path: str,
    player_boxes: list[tuple[float, float, float, float]],
    labels: list[str],
) -> dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Each team's (shirt, shorts) RGB color, read off the still: the median
    across the team's players of each player's median band color.

    Returns {} -- meaning "use the fixed palette" -- when a team has no
    players or when the two shirts would look alike on the mockup. A
    striped kit comes out as the blend of its stripes.
    """
    image = Image.open(image_path).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    colors = {}
    for team in (TEAM_A_LABEL, TEAM_B_LABEL):
        boxes = [box for box, label in zip(player_boxes, labels) if label == team]
        if not boxes:
            return {}
        shirt = np.median([_band_color(image_bgr, box, SHIRT_BAND) for box in boxes], axis=0)
        shorts = np.median([_band_color(image_bgr, box, SHORTS_BAND) for box in boxes], axis=0)
        colors[team] = (shirt, shorts)

    shirt_a, shirt_b = colors[TEAM_A_LABEL][0], colors[TEAM_B_LABEL][0]
    if np.linalg.norm(shirt_a - shirt_b) < KIT_MIN_DISTANCE:
        return {}

    return {
        team: tuple(tuple(int(round(channel)) for channel in color) for color in pair)
        for team, pair in colors.items()
    }
