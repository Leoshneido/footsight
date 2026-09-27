import cv2

WINDOW_NAME = "Click to place the ball, Delete to remove  --  Enter to accept, Esc to cancel"
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


ENTER_KEYS = (13, 10)
ESCAPE_KEY = 27
REMOVE_KEYS = (127, 8)  # Delete, Backspace


def handle_key(
    key: int,
    current: tuple[float, float] | None,
    detected: tuple[float, float] | None,
) -> tuple[bool, tuple[float, float] | None]:
    """What a key press does in the ball review window, as (done, ball):
    Enter accepts the ball as it now stands (including no ball), Esc
    throws away the edits and keeps what detection found, Delete or
    Backspace removes the ball."""
    if key in ENTER_KEYS:
        return True, current
    if key == ESCAPE_KEY:
        return True, detected
    if key in REMOVE_KEYS:
        return False, None
    return False, current


def review_ball(
    image_path: str,
    detected: tuple[float, float] | None,
    max_width_px: int = DEFAULT_MAX_WIDTH_PX,
) -> tuple[float, float] | None:
    """Show the still with the detected ball circled (if any) and let the
    user fix it: click to move it or add one, Delete/Backspace to remove
    it. Returns the ball in the still's own pixels, or None for no ball.

    The window loop itself is not unit-tested -- it needs a real display.
    handle_key and the coordinate helpers carry the logic that can go wrong.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Could not read {image_path} to review the ball on")

    height, width = image.shape[:2]
    scale = display_scale((width, height), max_width_px)
    preview = cv2.resize(image, None, fx=scale, fy=scale) if scale != 1.0 else image.copy()

    ball = [detected]

    def redraw() -> None:
        marked = preview.copy()
        if ball[0] is not None:
            x, y = ball[0]
            cv2.circle(marked, (int(x * scale), int(y * scale)), 10, (0, 0, 255), 2)
        cv2.imshow(WINDOW_NAME, marked)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ball[0] = to_source_pixel((float(x), float(y)), scale)
        redraw()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    redraw()

    try:
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key != 0xFF:
                done, ball[0] = handle_key(key, ball[0], detected)
                if done:
                    return ball[0]
                redraw()
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return ball[0]
    finally:
        cv2.destroyWindow(WINDOW_NAME)
