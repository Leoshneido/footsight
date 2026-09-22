import cv2
import numpy as np
from PIL import Image

TEAM_A_LABEL = "team_a"
TEAM_B_LABEL = "team_b"
OFFICIALS_LABEL = "officials"


def _jersey_color(image_bgr: np.ndarray, player_box: tuple[float, float, float, float]) -> float:
    """Median hue of a player's torso -- the middle third of their box
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
    pixels = torso_hsv.reshape(-1, 3).astype(np.float32)
    median = np.median(pixels, axis=0)
    return float(median[0])


def classify_players(
    image_path: str,
    player_boxes: list[tuple[float, float, float, float]],
) -> list[str]:
    """Label each player as "team_a", "team_b", or "officials" by clustering
    jersey colors into 3 groups. The two largest clusters are the teams; the
    smallest is the referee (a referee's kit is chosen to stand out from
    both teams, so it reliably forms its own small cluster).

    Falls back to labeling everyone "team_a" when there are too few players
    to meaningfully form 3 clusters.
    """
    if len(player_boxes) < 3:
        return [TEAM_A_LABEL] * len(player_boxes)

    image = Image.open(image_path).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    hues = np.array([[_jersey_color(image_bgr, box)] for box in player_boxes], dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, cluster_indices, _ = cv2.kmeans(hues, 3, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    cluster_indices = cluster_indices.flatten()

    counts = np.bincount(cluster_indices, minlength=3)
    officials_cluster = int(np.argmin(counts))
    team_clusters = [i for i in range(3) if i != officials_cluster]

    labels = []
    for cluster in cluster_indices:
        if cluster == officials_cluster:
            labels.append(OFFICIALS_LABEL)
        elif cluster == team_clusters[0]:
            labels.append(TEAM_A_LABEL)
        else:
            labels.append(TEAM_B_LABEL)
    return labels
