"""Generate the footsight logo: "FOOTSiGHT" in Kanit Black Italic, with a
classic football as the dot of the lowercase i.

The ball sits exactly where the font puts its own dot (so it follows the
italic lean), scaled up BALL_SCALE times with its bottom kept at the dot's
bottom. The ball is a real football pattern -- a truncated icosahedron
(black pentagons, white hexagons) projected onto a sphere so the seams curve.

Kanit is SIL Open Font License 1.1 (Copyright 2020 The Kanit Project
Authors); logos made with it carry no font-license restriction. The font
file is read from the user's font folder rather than copied into the repo.

It also writes the Chrome extension's toolbar icons: the logo's "i" with its
football dot, on the navy board color.

    python scripts/make_logo.py [--font PATH] [--out-dir assets/logo] [--icon-dir extension/icons]
"""
import argparse
import itertools
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull

DEFAULT_FONT = Path.home() / "Library" / "Fonts" / "Kanit-BlackItalic.ttf"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "logo"
DEFAULT_ICON_DIR = Path(__file__).resolve().parent.parent / "extension" / "icons"
ICON_SIZES = (16, 32, 48, 128)

WORD = ("FOOTS", "i", "GHT")
TEXT_COLOR = (255, 255, 255)
BOARD_COLOR = (18, 28, 48)  # navy, for the ad-board version
BALL_SCALE = 1.875  # ball diameter / the font's own dot height
BALL_WHITE = (250, 250, 250)
BALL_BLACK = (22, 22, 22)
BALL_SEAM = (60, 60, 60)
BALL_RIM = (255, 255, 255)
SUPERSAMPLE = 4

PHI = (1 + 5 ** 0.5) / 2


def _truncated_icosahedron() -> np.ndarray:
    base = [(0, 1, 3 * PHI), (1, 2 + PHI, 2 * PHI), (PHI, 2, PHI ** 3)]
    points = set()
    for a, b, c in base:
        for perm in ((a, b, c), (b, c, a), (c, a, b)):  # even permutations
            for signs in itertools.product((1, -1), repeat=3):
                points.add(tuple(round(s * x, 9) for s, x in zip(signs, perm)))
    return np.array(sorted(points))


def _faces(vertices: np.ndarray) -> list[tuple[np.ndarray, list[int]]]:
    """Each face as (outward normal, vertex indices in order around it)."""
    hull = ConvexHull(vertices)
    groups: dict[tuple, set] = {}
    for simplex, equation in zip(hull.simplices, hull.equations):
        groups.setdefault(tuple(np.round(equation[:3], 5)), set()).update(simplex)
    faces = []
    for normal, members in groups.items():
        members = list(members)
        normal = np.array(normal)
        center = vertices[members].mean(axis=0)
        u = vertices[members[0]] - center
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        angles = [math.atan2(np.dot(vertices[i] - center, v), np.dot(vertices[i] - center, u)) for i in members]
        faces.append((normal, [i for _, i in sorted(zip(angles, members))]))
    return faces


def _rotation_to_z(vector: np.ndarray) -> np.ndarray:
    vector = vector / np.linalg.norm(vector)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(vector, z)
    sine, cosine = np.linalg.norm(axis), np.dot(vector, z)
    if sine < 1e-9:
        return np.eye(3)
    axis /= sine
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + sine * k + (1 - cosine) * k @ k


def draw_ball(diameter_px: int, tilt_deg: tuple[float, float] = (18.0, -12.0)) -> Image.Image:
    """A classic black-and-white football, one pentagon roughly facing the
    viewer, tilted a little so it doesn't look like a flat badge."""
    size = diameter_px * SUPERSAMPLE
    radius = size / 2 * 0.97
    center = size / 2

    vertices = _truncated_icosahedron()
    faces = _faces(vertices)
    rotation = _rotation_to_z(next(n for n, idx in faces if len(idx) == 5))
    tilt_x, tilt_y = (math.radians(a) for a in tilt_deg)
    rot_x = np.array([[1, 0, 0], [0, math.cos(tilt_x), -math.sin(tilt_x)], [0, math.sin(tilt_x), math.cos(tilt_x)]])
    rot_y = np.array([[math.cos(tilt_y), 0, math.sin(tilt_y)], [0, 1, 0], [-math.sin(tilt_y), 0, math.cos(tilt_y)]])
    rotation = rot_y @ rot_x @ rotation
    rotated = vertices @ rotation.T

    def project(point: np.ndarray) -> tuple[float, float]:
        point = point / np.linalg.norm(point)  # onto the sphere: curved seams
        x, y, z = point
        if z < 0:  # behind the silhouette: pin to the rim instead of folding back inside
            flat = math.hypot(x, y) or 1.0
            x, y = x / flat, y / flat
        return center + x * radius, center - y * radius

    art = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(art)
    draw.ellipse([center - radius, center - radius, center + radius, center + radius], fill=BALL_WHITE)
    seam_width = max(1, int(radius * 0.025))
    for normal, members in sorted(faces, key=lambda f: (f[0] @ rotation.T)[2]):
        if (normal @ rotation.T)[2] <= -0.35:
            continue
        outline = []
        for a, b in zip(members, members[1:] + members[:1]):
            for t in np.linspace(0, 1, 12, endpoint=False):
                outline.append(project(rotated[a] * (1 - t) + rotated[b] * t))
        if len(members) == 5:
            draw.polygon(outline, fill=BALL_BLACK)
        else:
            draw.line(outline + outline[:1], fill=BALL_SEAM, width=seam_width)

    disc = Image.new("L", (size, size), 0)
    ImageDraw.Draw(disc).ellipse([center - radius, center - radius, center + radius, center + radius], fill=255)
    ball = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ball.paste(art, (0, 0), disc)
    ImageDraw.Draw(ball).ellipse(
        [center - radius, center - radius, center + radius, center + radius],
        outline=BALL_RIM, width=max(2, int(radius * 0.05)),
    )
    return ball.resize((diameter_px, diameter_px), Image.LANCZOS)


def _native_dot(font: ImageFont.FreeTypeFont, size: int) -> tuple[float, float, float]:
    """Center (x, y) of the font's own dot on 'i', relative to the letter's
    left-baseline anchor, and the dot's height."""
    probe = Image.new("L", (size * 4, size * 4), 0)
    origin = (size, size * 3)
    ImageDraw.Draw(probe).text(origin, "i", font=font, fill=255, anchor="ls")
    mask = np.array(probe) > 127
    rows = np.where(mask.any(axis=1))[0]
    last_dot_row = rows[np.where(np.diff(rows) > 1)[0][0]]
    ys, xs = np.nonzero(mask[: last_dot_row + 1])
    return xs.mean() - origin[0], ys.mean() - origin[1], ys.max() - ys.min() + 1


def make_logo(font_path: Path, cap_height_px: int) -> Image.Image:
    """The logo on a transparent background, cropped tight."""
    font_size = int(cap_height_px / 0.64)  # Kanit Black Italic: cap height ~0.64 em
    font = ImageFont.truetype(str(font_path), font_size)
    layer = Image.new("RGBA", (font_size * 12, font_size * 5), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x, baseline = font_size * 1.0, font_size * 3.5
    for part in WORD:
        if part == "i":
            draw.text((x, baseline), "ı", font=font, fill=TEXT_COLOR, anchor="ls")  # dotless i
            dot_dx, dot_dy, dot_height = _native_dot(font, font_size)
            diameter = int(round(dot_height * BALL_SCALE))
            ball = draw_ball(diameter)
            dot_bottom = baseline + dot_dy + dot_height / 2
            layer.paste(ball, (int(round(x + dot_dx - diameter / 2)), int(round(dot_bottom - diameter))), ball)
        else:
            draw.text((x, baseline), part, font=font, fill=TEXT_COLOR, anchor="ls")
        x += draw.textlength(part, font=font)
    return layer.crop(layer.getbbox())


def make_icon(font_path: Path, size: int) -> Image.Image:
    """The extension icon: the logo's i and football dot, drawn exactly as in
    the logo, centered on a rounded navy tile."""
    big = 1024
    font = ImageFont.truetype(str(font_path), big)
    layer = Image.new("RGBA", (big * 3, big * 3), (0, 0, 0, 0))
    x, baseline = big * 1.0, big * 2.5
    ImageDraw.Draw(layer).text((x, baseline), "ı", font=font, fill=TEXT_COLOR, anchor="ls")
    dot_dx, dot_dy, dot_height = _native_dot(font, big)
    diameter = int(round(dot_height * BALL_SCALE))
    ball = draw_ball(diameter)
    dot_bottom = baseline + dot_dy + dot_height / 2
    layer.paste(ball, (int(round(x + dot_dx - diameter / 2)), int(round(dot_bottom - diameter))), ball)
    mark = layer.crop(layer.getbbox())

    tile_px = size * SUPERSAMPLE
    tile = Image.new("RGBA", (tile_px, tile_px), (0, 0, 0, 0))
    ImageDraw.Draw(tile).rounded_rectangle((0, 0, tile_px - 1, tile_px - 1), radius=int(tile_px * 0.22), fill=BOARD_COLOR + (255,))
    scale = tile_px * 0.78 / max(mark.size)
    mark = mark.resize((max(1, int(mark.width * scale)), max(1, int(mark.height * scale))), Image.LANCZOS)
    tile.alpha_composite(mark, ((tile_px - mark.width) // 2, (tile_px - mark.height) // 2))
    return tile.resize((size, size), Image.LANCZOS)


def on_board(logo: Image.Image, padding_ratio: float = 0.25) -> Image.Image:
    pad_x, pad_y = int(logo.width * padding_ratio / 4), int(logo.height * padding_ratio)
    board = Image.new("RGB", (logo.width + 2 * pad_x, logo.height + 2 * pad_y), BOARD_COLOR)
    board.paste(logo, (pad_x, pad_y), logo)
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--icon-dir", type=Path, default=DEFAULT_ICON_DIR)
    parser.add_argument("--cap-height", type=int, default=400, help="Height of the capitals in pixels")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logo = make_logo(args.font, args.cap_height)
    logo.save(args.out_dir / "footsight_logo.png")
    on_board(logo).save(args.out_dir / "footsight_logo_board.png")
    draw_ball(512).save(args.out_dir / "footsight_ball.png")
    print(f"Wrote logo files to {args.out_dir}")
    args.icon_dir.mkdir(parents=True, exist_ok=True)
    for size in ICON_SIZES:
        make_icon(args.font, size).save(args.icon_dir / f"icon{size}.png")
    print(f"Wrote extension icons to {args.icon_dir}")


if __name__ == "__main__":
    main()
