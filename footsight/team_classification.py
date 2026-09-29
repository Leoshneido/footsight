import cv2
import numpy as np
from PIL import Image

TEAM_A_LABEL = "team_a"
TEAM_B_LABEL = "team_b"
OFFICIALS_LABEL = "officials"

# How far (in hue units, on the 0-179 circle) a player's jersey can be from
# their team's color before they're set aside as officials. On the sample
# stills: Barca's referee sits 61-70 from both teams, a striped Barca shirt
# 27 from its team, Portugal's red within ~12 of its team.
STRAY_HUE_DISTANCE = 40.0

# The pitch showing through a player's crop is left out of their jersey
# color: a stretched, diving or blurred player fills only part of the box,
# and the grass around them can outvote the jersey. The grass hue differs
# per broadcast (~34 Bayern, ~39 Colombia v Portugal, ~44 Barca), so it is
# measured on each still, beside the players, and pixels within
# GRASS_HUE_TOLERANCE of it are dropped. DEFAULT_GRASS_HUE is the fallback
# when there's no grass to measure.
GRASS_HUE_TOLERANCE = 6.0
GRASS_MIN_SATURATION = 60  # pitch lines and white kits aren't grass samples
DEFAULT_GRASS_HUE = 50.0

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


def estimate_grass_hue(image_bgr: np.ndarray, player_boxes) -> float:
    """The pitch's hue on this still: the circular median of saturated
    pixels in strips just left and right of each player, at torso height."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    samples = []
    for box in player_boxes:
        x1, y1, x2, y2 = (int(v) for v in box)
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        for xa, xb in ((x1 - w, x1 - w // 4), (x2 + w // 4, x2 + w)):
            patch = hsv[max(0, y1 + h // 3):max(0, y1 + 2 * h // 3), max(0, xa):max(0, xb)].reshape(-1, 3)
            samples.append(patch[patch[:, 1] >= GRASS_MIN_SATURATION, 0])
    hues = np.concatenate(samples).astype(np.float32) if samples else np.array([], np.float32)
    return _circular_median(hues) if hues.size else DEFAULT_GRASS_HUE


def _not_grass(hues: np.ndarray, grass_hue: float) -> np.ndarray:
    d = np.abs(hues.astype(np.float32) - grass_hue) % HUE_PERIOD
    return np.minimum(d, HUE_PERIOD - d) > GRASS_HUE_TOLERANCE


def _jersey_hues(image_bgr: np.ndarray, player_box: tuple[float, float, float, float],
                 grass_hue: float = DEFAULT_GRASS_HUE) -> np.ndarray:
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
    not_grass = hues[_not_grass(hues, grass_hue)]
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


def _circular_distance(a, b):
    """Distance between hues on the hue circle (0-179 wraps around)."""
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % HUE_PERIOD
    return np.minimum(d, HUE_PERIOD - d)


def _fit_two_teams(hues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """k-means (k=2) on the hue circle: each hue a point on a unit circle,
    so 179 and 0 sit next to each other. Returns (cluster per hue, the two
    cluster centers as hues)."""
    angles = hues * (2 * np.pi / HUE_PERIOD)
    points = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    cv2.setRNGSeed(0)  # same still, same answer
    _, clusters, centers = cv2.kmeans(points, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    center_hues = (np.degrees(np.arctan2(centers[:, 1], centers[:, 0])) % 360) / 2
    return clusters.flatten(), center_hues


def classify_players(
    image_path: str,
    player_boxes: list[tuple[float, float, float, float]],
) -> list[str]:
    """Label each player "team_a", "team_b" or "officials" by clustering
    jersey hues into 2 team groups on the hue circle.

    Mostly outfield players reach this function -- the detector tags
    goalkeepers and officials with their own roles -- which is what lets
    this fit exactly 2 clusters. But the detector sometimes hands a referee
    (or a watermark) over as a player. So the two teams are fitted, anyone
    further than STRAY_HUE_DISTANCE from their team's color is set aside as
    officials, and the teams are refitted without them, until nothing
    changes. A cluster left with a single member is a stray too: one odd
    color must not claim a whole team.

    Everything is on the hue circle (circular medians, circular distances):
    red kits read ~175 on some players and ~3 on others, the same color
    either side of the seam. The earlier straight-line trim threw most of
    Portugal out as officials on the Colombia v Portugal captures; a Barca
    striped kit (claret either side of the seam plus blue) broke it before.

    Falls back to labeling everyone "team_a" when there are too few players
    to meaningfully form 2 clusters.
    """
    if len(player_boxes) < 2:
        return [TEAM_A_LABEL] * len(player_boxes)

    image = Image.open(image_path).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    grass_hue = estimate_grass_hue(image_bgr, player_boxes)
    hues = np.array([_circular_median(_jersey_hues(image_bgr, box, grass_hue)) for box in player_boxes], dtype=np.float32)

    team = list(range(len(player_boxes)))
    for _ in range(len(player_boxes)):
        if len(team) < 2:
            return [TEAM_A_LABEL] * len(player_boxes)
        clusters, centers = _fit_two_teams(hues[team])
        distance = _circular_distance(hues[team], centers[clusters])
        strays = {team[i] for i in np.flatnonzero(distance > STRAY_HUE_DISTANCE)}
        if len(team) > 3:
            for cluster in (0, 1):
                members = [team[i] for i in np.flatnonzero(clusters == cluster)]
                if len(members) == 1:
                    strays.update(members)
        if not strays:
            break
        team = [index for index in team if index not in strays]

    first_cluster = int(clusters[0])
    labels = [OFFICIALS_LABEL] * len(player_boxes)
    for position, index in enumerate(team):
        labels[index] = TEAM_A_LABEL if clusters[position] == first_cluster else TEAM_B_LABEL
    return labels


def _band_color(
    image_bgr: np.ndarray,
    player_box: tuple[float, float, float, float],
    band: tuple[float, float],
    grass_hue: float = DEFAULT_GRASS_HUE,
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
    not_grass = pixels[_not_grass(hues, grass_hue)]
    if len(not_grass) > 0:
        pixels = not_grass
    blue, green, red = np.median(pixels, axis=0)
    return np.array([red, green, blue])


def official_kits(image_path: str, boxes) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Each official's own (shirt, shorts) RGB, read like team kits (median
    of the shirt and shorts bands, grass left out) but per person: officials
    are drawn in what they actually wear, not a fixed color."""
    if not boxes:
        return []
    image = Image.open(image_path).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    grass_hue = estimate_grass_hue(image_bgr, boxes)
    kits = []
    for box in boxes:
        shirt = _band_color(image_bgr, box, SHIRT_BAND, grass_hue)
        shorts = _band_color(image_bgr, box, SHORTS_BAND, grass_hue)
        kits.append((tuple(int(round(c)) for c in shirt), tuple(int(round(c)) for c in shorts)))
    return kits


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

    grass_hue = estimate_grass_hue(image_bgr, player_boxes)
    colors = {}
    for team in (TEAM_A_LABEL, TEAM_B_LABEL):
        boxes = [box for box, label in zip(player_boxes, labels) if label == team]
        if not boxes:
            return {}
        shirt = np.median([_band_color(image_bgr, box, SHIRT_BAND, grass_hue) for box in boxes], axis=0)
        shorts = np.median([_band_color(image_bgr, box, SHORTS_BAND, grass_hue) for box in boxes], axis=0)
        colors[team] = (shirt, shorts)

    shirt_a, shirt_b = colors[TEAM_A_LABEL][0], colors[TEAM_B_LABEL][0]
    if np.linalg.norm(shirt_a - shirt_b) < KIT_MIN_DISTANCE:
        return {}

    return {
        team: tuple(tuple(int(round(channel)) for channel in color) for color in pair)
        for team, pair in colors.items()
    }
