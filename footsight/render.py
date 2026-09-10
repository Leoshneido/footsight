from PIL import Image, ImageDraw

PITCH_COLOR = (34, 139, 34)
LINE_COLOR = (255, 255, 255)
PLAYER_COLOR = (220, 20, 60)


def pitch_to_image_coords(
    point: tuple[float, float],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    image_width_px: int = 1050,
    margin_px: int = 40,
) -> tuple[float, float]:
    x, y = point
    scale = (image_width_px - 2 * margin_px) / pitch_length
    px = margin_px + (x + pitch_length / 2.0) * scale
    py = margin_px + (y + pitch_width / 2.0) * scale
    return px, py


def render_pitch(
    player_positions: list[tuple[float, float]],
    output_path: str,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    image_width_px: int = 1050,
    margin_px: int = 40,
    dot_radius_px: int = 8,
) -> None:
    scale = (image_width_px - 2 * margin_px) / pitch_length
    image_height_px = int(pitch_width * scale) + 2 * margin_px

    image = Image.new("RGB", (image_width_px, image_height_px), PITCH_COLOR)
    draw = ImageDraw.Draw(image)

    top_left = pitch_to_image_coords(
        (-pitch_length / 2, -pitch_width / 2), pitch_length, pitch_width, image_width_px, margin_px
    )
    bottom_right = pitch_to_image_coords(
        (pitch_length / 2, pitch_width / 2), pitch_length, pitch_width, image_width_px, margin_px
    )
    draw.rectangle([top_left, bottom_right], outline=LINE_COLOR, width=2)

    halfway_top = pitch_to_image_coords((0.0, -pitch_width / 2), pitch_length, pitch_width, image_width_px, margin_px)
    halfway_bottom = pitch_to_image_coords((0.0, pitch_width / 2), pitch_length, pitch_width, image_width_px, margin_px)
    draw.line([halfway_top, halfway_bottom], fill=LINE_COLOR, width=2)

    for position in player_positions:
        px, py = pitch_to_image_coords(position, pitch_length, pitch_width, image_width_px, margin_px)
        draw.ellipse(
            [px - dot_radius_px, py - dot_radius_px, px + dot_radius_px, py + dot_radius_px],
            fill=PLAYER_COLOR,
        )

    image.save(output_path)
