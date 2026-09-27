import cv2

from footsight.ball_picker import DEFAULT_MAX_WIDTH_PX, display_scale, to_source_pixel

WINDOW_NAME = "Click detections to remove  --  Enter to accept, Esc to cancel"

Detection = tuple[tuple[float, float, float, float], str]


def detection_at(point: tuple[float, float], detections: list[Detection]) -> int | None:
    """Index of the detection whose box contains the point, or None. Where
    boxes overlap, the smallest wins -- that's the one the click is aimed at."""
    x, y = point
    hits = [
        index
        for index, ((x1, y1, x2, y2), _) in enumerate(detections)
        if x1 <= x <= x2 and y1 <= y <= y2
    ]
    if not hits:
        return None

    def area(index: int) -> float:
        x1, y1, x2, y2 = detections[index][0]
        return (x2 - x1) * (y2 - y1)

    return min(hits, key=area)


def drop_detections(detections: list[Detection], removed: set[int]) -> list[Detection]:
    return [detection for index, detection in enumerate(detections) if index not in removed]


def review_detections(
    image_path: str,
    detections: list[Detection],
    max_width_px: int = DEFAULT_MAX_WIDTH_PX,
) -> list[Detection]:
    """Show the still with every detection boxed and let the user click the
    false ones (a broadcaster watermark, a ball boy) to remove them. Clicking
    a removed box again restores it. Esc keeps every detection.

    The window loop itself is not unit-tested -- it needs a real display.
    detection_at and drop_detections carry the logic that can go wrong.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Could not read {image_path} to review detections on")

    height, width = image.shape[:2]
    scale = display_scale((width, height), max_width_px)
    preview = cv2.resize(image, None, fx=scale, fy=scale) if scale != 1.0 else image.copy()

    removed: set[int] = set()

    def redraw() -> None:
        marked = preview.copy()
        for index, ((x1, y1, x2, y2), _) in enumerate(detections):
            top_left = (int(x1 * scale), int(y1 * scale))
            bottom_right = (int(x2 * scale), int(y2 * scale))
            if index in removed:
                cv2.rectangle(marked, top_left, bottom_right, (0, 0, 255), 2)
                cv2.line(marked, top_left, bottom_right, (0, 0, 255), 2)
                cv2.line(marked, (top_left[0], bottom_right[1]), (bottom_right[0], top_left[1]), (0, 0, 255), 2)
            else:
                cv2.rectangle(marked, top_left, bottom_right, (0, 255, 0), 1)
        cv2.imshow(WINDOW_NAME, marked)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        index = detection_at(to_source_pixel((float(x), float(y)), scale), detections)
        if index is None:
            return
        removed.symmetric_difference_update({index})
        redraw()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    redraw()

    try:
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == 27:  # Esc
                return detections
            if key in (13, 10):  # Enter
                return drop_detections(detections, removed)
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return drop_detections(detections, removed)
    finally:
        cv2.destroyWindow(WINDOW_NAME)
