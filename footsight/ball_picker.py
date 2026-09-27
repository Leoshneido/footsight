import cv2

WINDOW_NAME = "Click the ball  --  Enter to accept, Esc to cancel"
DEFAULT_MAX_WIDTH_PX = 1400


def display_scale(image_size: tuple[int, int], max_width_px: int = DEFAULT_MAX_WIDTH_PX) -> float:
    """How far a still must shrink to fit on screen. Broadcast stills are
    ~3000px wide; anything already narrower is shown at full size."""
    width, _ = image_size
    if width <= max_width_px:
        return 1.0
    return max_width_px / width


def to_source_pixel(click: tuple[float, float], scale: float) -> tuple[float, float]:
    """Map a click in the shrunken preview back to the still's own pixels,
    which is the coordinate space the homography projects from."""
    x, y = click
    return (x / scale, y / scale)


def pick_ball_pixel(
    image_path: str,
    max_width_px: int = DEFAULT_MAX_WIDTH_PX,
) -> tuple[float, float] | None:
    """Show the still and let the user click where the ball is. Returns the
    clicked point in the still's own pixels, or None if they cancelled.

    The window loop itself is not unit-tested -- it needs a real display.
    The two coordinate helpers above carry the logic that can go wrong.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Could not read {image_path} to pick the ball from")

    height, width = image.shape[:2]
    scale = display_scale((width, height), max_width_px)
    preview = cv2.resize(image, None, fx=scale, fy=scale) if scale != 1.0 else image.copy()
    clean = preview.copy()

    clicked: list[tuple[float, float]] = []

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        clicked.clear()
        clicked.append((float(x), float(y)))
        marked = clean.copy()
        cv2.circle(marked, (x, y), 10, (0, 0, 255), 2)
        cv2.imshow(WINDOW_NAME, marked)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    cv2.imshow(WINDOW_NAME, preview)

    try:
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == 27:  # Esc
                return None
            if key in (13, 10):  # Enter
                return to_source_pixel(clicked[0], scale) if clicked else None
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return to_source_pixel(clicked[0], scale) if clicked else None
    finally:
        cv2.destroyWindow(WINDOW_NAME)
